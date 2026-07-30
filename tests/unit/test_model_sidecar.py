"""Trained-model sidecar and the post-training lifecycle (issue #74).

Training used to end at a message box saying `Training complete`, after which
the model you had just trained was not selected for prediction, lived wherever
Ultralytics put it, and told you nothing about whether it was any good. The
manual reload in the middle of that was the sharpest edge in the workflow.

The sidecar tests carry a specific promise: reading one must be a strict
improvement and never a requirement, so a model trained outside the app keeps
loading through the bare-``kpt_shape`` path exactly as before.
"""

import json
import os
import subprocess
import sys

import pytest
from PyQt6.QtWidgets import QWidget

from src.digitalsreeni_image_annotator.controllers.model_registry_controller import (
    ModelRegistryController,
)
from src.digitalsreeni_image_annotator.core import model_sidecar


# --- Qt-free ---------------------------------------------------------------


def test_the_sidecar_module_imports_without_qt():
    """The CLI (#76) reads sidecars; it cannot require a display."""
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.model_sidecar as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- sidecar payload -------------------------------------------------------


def test_sidecar_path_sits_next_to_the_weights(tmp_path):
    weights = tmp_path / "best_2026.pt"
    sidecar = model_sidecar.sidecar_path(str(weights))
    assert os.path.dirname(sidecar) == str(tmp_path)
    assert os.path.basename(sidecar) == "best_2026.json"


def test_absent_values_are_omitted_not_written_as_null():
    """A reader distinguishing "not applicable" from "recorded as nothing" has
    an easier job, and the file stays human-readable."""
    payload = model_sidecar.build_sidecar(model_type="yolo")
    assert payload == {"schema_version": 1, "model_type": "yolo"}
    assert "keypoint_schema" not in payload
    assert "metrics" not in payload


def test_sidecar_round_trips(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    payload = model_sidecar.build_sidecar(
        model_type="yolo",
        task="pose",
        class_names=["person"],
        kpt_shape=[17, 3],
        flip_idx=list(range(17)),
        metrics={"mAP50": 0.83},
        config={"epochs": 100},
        timestamp="20260725-101500",
    )
    model_sidecar.write_sidecar(str(weights), payload)

    loaded = model_sidecar.read_sidecar(str(weights))
    assert loaded["task"] == "pose"
    assert loaded["kpt_shape"] == [17, 3]
    assert loaded["flip_idx"] == list(range(17))
    assert loaded["metrics"]["mAP50"] == pytest.approx(0.83)


def test_a_missing_sidecar_reads_as_none(tmp_path):
    """The fallback path externally-trained models rely on."""
    weights = tmp_path / "external.pt"
    weights.write_bytes(b"weights")
    assert model_sidecar.read_sidecar(str(weights)) is None


def test_a_corrupt_sidecar_is_no_worse_than_none(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    (tmp_path / "model.json").write_text("{not json", encoding="utf-8")
    assert model_sidecar.read_sidecar(str(weights)) is None


def test_a_sidecar_that_is_not_an_object_reads_as_none(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    (tmp_path / "model.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert model_sidecar.read_sidecar(str(weights)) is None


# --- collision handling ----------------------------------------------------


def test_a_name_collision_does_not_overwrite(tmp_path):
    """Two runs finishing inside the same second is entirely possible on a fast
    machine with a small dataset, and the failure mode is destroying a model
    the user may not have copied anywhere yet."""
    first = model_sidecar.unique_weights_path(str(tmp_path), "yolo_best", "20260725")
    open(first, "wb").close()
    second = model_sidecar.unique_weights_path(str(tmp_path), "yolo_best", "20260725")
    assert second != first
    assert not os.path.exists(second)


def test_unique_weights_path_creates_the_directory(tmp_path):
    target = model_sidecar.unique_weights_path(
        str(tmp_path / "models"), "m", "20260725"
    )
    assert os.path.isdir(os.path.dirname(target))


# --- metric formatting -----------------------------------------------------


def test_only_reported_metrics_are_formatted():
    """An empty row reads as "the model scored nothing", a very different claim
    from "this path does not report that"."""
    rows = model_sidecar.format_metrics(
        {"mAP50": 0.8123, "recall": 0.77, "unknown_key": 1}
    )
    labels = [label for label, _ in rows]
    assert labels == ["mAP@50", "Recall"]
    assert dict(rows)["mAP@50"] == "0.8123"


def test_an_unproduced_metric_has_no_label_mapping():
    """epochs_completed was mapped but never populated — train_model returns
    Ultralytics' metrics object, which has no `.trainer` to read an epoch from.
    A label for a metric nothing emits is a row that pretends to try."""
    assert model_sidecar.format_metrics({"epochs_completed": 40}) == []


def test_formatting_no_metrics_yields_no_rows():
    assert model_sidecar.format_metrics({}) == []
    assert model_sidecar.format_metrics(None) == []


# --- the post-training routine --------------------------------------------


class _FakeWindow(QWidget):
    def __init__(self, project_dir=None, image=""):
        super().__init__()
        if project_dir is not None:
            self.current_project_dir = project_dir
        self.is_loading_project = False
        self.class_mapping = {"cell": 1, "Temp-auto": 2}
        self.keypoint_schemas = {}
        self.image_file_name = image
        self.predicted = []

    def predict_single_image(self, file_name):
        self.predicted.append(file_name)


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "runs" / "weights"
    path.mkdir(parents=True)
    best = path / "best.pt"
    best.write_bytes(b"fake weights")
    return str(best)


def _registry(qtbot, project_dir, image=""):
    window = _FakeWindow(project_dir, image)
    qtbot.addWidget(window)
    return ModelRegistryController(window), window


def test_a_successful_run_saves_weights_and_a_sidecar(qtbot, tmp_path, weights):
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))

    summary = registry.finish_run(
        model_type="yolo", result={"ok": True}, weights_path=weights,
        metrics={"mAP50": 0.9}, config={"epochs": 10},
    )

    assert summary is not None
    assert os.path.exists(summary["weights_path"])
    assert os.path.exists(summary["sidecar_path"])
    with open(summary["sidecar_path"], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["class_names"] == ["cell"], "Temp-* classes are not real classes"
    assert payload["metrics"]["mAP50"] == pytest.approx(0.9)


def test_the_saved_name_is_not_just_best_pt(qtbot, tmp_path, weights):
    """Ultralytics names every run's output best.pt, which tells the user
    nothing once several runs share a directory."""
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))
    summary = registry.finish_run(
        model_type="yolo", result={}, weights_path=weights
    )
    name = os.path.basename(summary["weights_path"])
    assert name.startswith("yolo_best_")
    assert name != "best.pt"


def test_a_failed_run_registers_and_writes_nothing(qtbot, tmp_path, weights):
    """Both trainers hand back a result that can represent an error;
    registering on one would put a broken model in front of the user as though
    it were ready."""
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))

    assert registry.finish_run(
        model_type="yolo", result="CUDA out of memory", weights_path=weights
    ) is None
    assert not os.path.isdir(project / "models")


def test_a_run_with_no_weights_registers_nothing(qtbot, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))
    assert registry.finish_run(
        model_type="yolo", result={}, weights_path=None
    ) is None
    assert registry.finish_run(
        model_type="yolo", result={}, weights_path=str(project / "nope.pt")
    ) is None


def test_nothing_is_written_while_a_project_is_loading(qtbot, tmp_path, weights):
    """Autosave and project writes are suspended during a load (ADR-005);
    writing a model into the project mid-load is the same hazard."""
    project = tmp_path / "project"
    project.mkdir()
    registry, window = _registry(qtbot, str(project))
    window.is_loading_project = True

    assert registry.finish_run(
        model_type="yolo", result={}, weights_path=weights
    ) is None
    assert not os.path.isdir(project / "models")


def test_without_a_project_the_weights_are_left_where_they_are(qtbot, weights):
    registry, _window = _registry(qtbot, None)
    summary = registry.finish_run(
        model_type="yolo", result={}, weights_path=weights
    )
    assert summary["weights_path"] == weights
    assert summary["sidecar_path"] is None


def test_two_runs_do_not_overwrite_each_other(qtbot, tmp_path, weights):
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))

    first = registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    second = registry.finish_run(model_type="yolo", result={}, weights_path=weights)

    assert first["weights_path"] != second["weights_path"]
    assert os.path.exists(first["weights_path"])


# --- keypoint schema carry-over (ADR-029 PR-3) -----------------------------


def test_a_shared_schema_is_recorded(qtbot, tmp_path, weights):
    project = tmp_path / "project"
    project.mkdir()
    registry, window = _registry(qtbot, str(project))
    schema = {"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]}
    window.class_mapping = {"person": 1}
    window.keypoint_schemas = {"person": schema}

    summary = registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    payload = model_sidecar.read_sidecar(summary["weights_path"])
    assert payload["keypoint_schema"] == schema


def test_disagreeing_schemas_record_none_rather_than_one_of_them(
    qtbot, tmp_path, weights
):
    """Recording one class's schema for a mixed project would be worse than
    recording none — the bare-kpt_shape reconstruction handles that case."""
    project = tmp_path / "project"
    project.mkdir()
    registry, window = _registry(qtbot, str(project))
    window.class_mapping = {"person": 1, "hand": 2}
    window.keypoint_schemas = {
        "person": {"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]},
        "hand": {"names": ["a"], "skeleton": [], "flip_idx": [0]},
    }

    summary = registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    payload = model_sidecar.read_sidecar(summary["weights_path"])
    assert "keypoint_schema" not in payload


def test_a_partially_schemad_project_records_none(qtbot, tmp_path, weights):
    project = tmp_path / "project"
    project.mkdir()
    registry, window = _registry(qtbot, str(project))
    window.class_mapping = {"person": 1, "cell": 2}
    window.keypoint_schemas = {
        "person": {"names": ["a"], "skeleton": [], "flip_idx": [0]}
    }

    summary = registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    payload = model_sidecar.read_sidecar(summary["weights_path"])
    assert "keypoint_schema" not in payload


# --- try it now ------------------------------------------------------------


def test_try_now_is_disabled_without_an_open_image(qtbot, tmp_path, weights):
    registry, _window = _registry(qtbot, str(tmp_path), image="")
    registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    assert registry.can_try_now() is False
    registry.try_on_current_image()  # must be a silent no-op


def test_try_now_runs_the_model_on_the_current_image(qtbot, tmp_path, weights):
    registry, window = _registry(qtbot, str(tmp_path), image="a.png")
    registry.finish_run(model_type="yolo", result={}, weights_path=weights)
    assert registry.can_try_now() is True
    registry.try_on_current_image()
    assert window.predicted == ["a.png"]


def test_try_now_is_unavailable_after_a_sam_fine_tune(qtbot, tmp_path, weights):
    """predict_single_image routes to the YOLO trainer regardless of what was
    trained, so offering it after a SAM run would execute the loaded YOLO model
    — or pop "No Model" — while the panel claimed the SAM model was active.
    A fine-tuned SAM checkpoint is used interactively via SAM-box/SAM-points.
    """
    registry, window = _registry(qtbot, str(tmp_path), image="a.png")
    registry.finish_run(model_type="sam", result={}, weights_path=weights)

    assert registry.can_try_now() is False
    registry.try_on_current_image()
    assert window.predicted == []


def test_try_now_is_unavailable_before_any_run(qtbot, tmp_path):
    registry, _window = _registry(qtbot, str(tmp_path), image="a.png")
    assert registry.can_try_now() is False


# --- disk usage is reported, not silently managed --------------------------


def test_the_models_directory_size_is_reported(qtbot, tmp_path, weights):
    project = tmp_path / "project"
    project.mkdir()
    registry, _window = _registry(qtbot, str(project))
    registry.finish_run(model_type="yolo", result={}, weights_path=weights)

    assert registry.models_dir_size_mb() is not None
    assert registry.models_dir_size_mb() >= 0


def test_the_size_is_none_before_anything_is_saved(qtbot, tmp_path):
    registry, _window = _registry(qtbot, str(tmp_path))
    assert registry.models_dir_size_mb() is None
