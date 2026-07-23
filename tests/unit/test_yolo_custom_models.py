"""Unit tests for the trained-YOLO-model registry + MLflow link wiring (#83).

Covers the pure/headless pieces of the in-app YOLO trainer that route a run
into ``models/yolo/custom/<name>`` and expose it for prediction, plus the
once-per-run MLflow deep-link emission. The full ``train_model`` needs a real
Ultralytics model, so these exercise the helpers around it directly.
"""

import types

import pytest
import yaml as yaml_lib

from src.digitalsreeni_image_annotator.dialogs import yolo_trainer as yt
from src.digitalsreeni_image_annotator.dialogs.yolo_trainer import (
    LoadPredictionModelDialog,
    YOLOTrainer,
    _sanitize_run_name,
    list_custom_yolo_models,
)


# --- _sanitize_run_name -----------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("MyProject", "MyProject"),
    ("My Proj #1!", "My_Proj__1"),
    ("a/b\\c", "a_b_c"),
    ("", "model"),
    ("___", "model"),
    ("keep-_chars", "keep-_chars"),
])
def test_sanitize_run_name(raw, expected):
    assert _sanitize_run_name(raw) == expected


# --- list_custom_yolo_models ------------------------------------------------

def _make_run(custom_dir, name, with_yaml=True, with_best=True):
    run = custom_dir / name
    (run / "weights").mkdir(parents=True)
    if with_best:
        (run / "weights" / "best.pt").write_bytes(b"fake")
    if with_yaml:
        (run / "data.yaml").write_text(yaml_lib.dump({"names": {0: "cell"}, "nc": 1}))
    return run


def test_list_custom_models_discovers_runs(tmp_path, monkeypatch):
    custom = tmp_path / "custom"
    custom.mkdir()
    _make_run(custom, "runA")
    _make_run(custom, "runB")
    monkeypatch.setattr(yt, "YOLO_CUSTOM_DIR", str(custom))

    listed = list_custom_yolo_models()
    assert set(listed) == {"★ runA", "★ runB"}
    for model_path, yaml_path in listed.values():
        assert model_path.endswith("best.pt")
        assert yaml_path.endswith("data.yaml")


def test_list_skips_runs_without_best(tmp_path, monkeypatch):
    custom = tmp_path / "custom"
    custom.mkdir()
    _make_run(custom, "good")
    _make_run(custom, "no_weights", with_best=False)
    monkeypatch.setattr(yt, "YOLO_CUSTOM_DIR", str(custom))

    listed = list_custom_yolo_models()
    assert set(listed) == {"★ good"}  # a run with no best.pt is not selectable


def test_list_missing_yaml_yields_empty_yaml_path(tmp_path, monkeypatch):
    custom = tmp_path / "custom"
    custom.mkdir()
    _make_run(custom, "noyaml", with_yaml=False)
    monkeypatch.setattr(yt, "YOLO_CUSTOM_DIR", str(custom))

    (_model, yaml_path), = list_custom_yolo_models().values()
    assert yaml_path == ""  # browse-for-yaml fallback, never a bogus path


def test_list_empty_when_dir_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(yt, "YOLO_CUSTOM_DIR", str(tmp_path / "does_not_exist"))
    assert list_custom_yolo_models() == {}


# --- _register_trained_model ------------------------------------------------

def _trainer(tmp_path, qapp):
    t = YOLOTrainer(str(tmp_path / "proj"), main_window=None)
    return t


def test_register_trained_model_writes_yaml_and_paths(tmp_path, qapp):
    # Fake Ultralytics model+trainer: best.pt on disk, names dict.
    run = tmp_path / "models" / "yolo" / "custom" / "proj"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"fake")

    t = _trainer(tmp_path, qapp)
    t.model = types.SimpleNamespace(
        names={0: "cell", 1: "nucleus"},
        trainer=types.SimpleNamespace(best=str(best), save_dir=str(run)),
    )
    t._register_trained_model()

    assert t.last_saved_model_path == str(best)
    data_yaml = run / "data.yaml"
    assert t.last_saved_yaml_path == str(data_yaml)
    written = yaml_lib.safe_load(data_yaml.read_text())
    assert written["nc"] == 2
    assert written["names"] == {0: "cell", 1: "nucleus"}


def test_register_falls_back_to_save_dir_when_best_attr_missing(tmp_path, qapp):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"fake")

    t = _trainer(tmp_path, qapp)
    # No usable .best, but save_dir points at the run -> best is derived.
    t.model = types.SimpleNamespace(
        names={0: "cell"},
        trainer=types.SimpleNamespace(best="", save_dir=str(run)),
    )
    t._register_trained_model()
    assert t.last_saved_model_path == str(best)


def test_register_noop_when_no_checkpoint(tmp_path, qapp):
    t = _trainer(tmp_path, qapp)
    t.model = types.SimpleNamespace(
        names={0: "cell"},
        trainer=types.SimpleNamespace(best="", save_dir=str(tmp_path / "empty")),
    )
    t._register_trained_model()
    assert t.last_saved_model_path is None
    assert t.last_saved_yaml_path is None


# --- _register_trained_model: pose metadata (#35 PR-3) ---------------------

def test_register_carries_kpt_shape_and_flip_idx(tmp_path, qapp):
    run = tmp_path / "models" / "yolo" / "custom" / "pose_proj"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"fake")

    train_yaml_path = tmp_path / "train_data.yaml"
    train_yaml_path.write_text(yaml_lib.dump({
        "names": {0: "person"},
        "nc": 1,
        "kpt_shape": [3, 3],
        "flip_idx": [0, 2, 1],
    }))

    t = _trainer(tmp_path, qapp)
    # No matching main_window.keypoint_schemas entries -> keypoint_schema key
    # must be omitted, but kpt_shape/flip_idx still carry over unconditionally.
    t.main_window = types.SimpleNamespace(keypoint_schemas={})
    t.yaml_path = str(train_yaml_path)
    t.model = types.SimpleNamespace(
        names={0: "person"},
        trainer=types.SimpleNamespace(best=str(best), save_dir=str(run)),
    )
    t._register_trained_model()

    written = yaml_lib.safe_load((run / "data.yaml").read_text())
    assert written["kpt_shape"] == [3, 3]
    assert written["flip_idx"] == [0, 2, 1]
    assert "keypoint_schema" not in written


def test_register_embeds_keypoint_schema_when_all_classes_match(tmp_path, qapp):
    run = tmp_path / "models" / "yolo" / "custom" / "pose_proj2"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"fake")

    train_yaml_path = tmp_path / "train_data2.yaml"
    train_yaml_path.write_text(yaml_lib.dump({
        "names": {0: "person", 1: "robot"},
        "nc": 2,
        "kpt_shape": [3, 3],
        "flip_idx": [0, 2, 1],
    }))

    schema = {
        "names": ["nose", "left_eye", "right_eye"],
        "skeleton": [[0, 1], [0, 2]],
        "flip_idx": [0, 2, 1],
    }
    t = _trainer(tmp_path, qapp)
    t.main_window = types.SimpleNamespace(
        keypoint_schemas={"person": schema, "robot": dict(schema)}
    )
    t.yaml_path = str(train_yaml_path)
    t.model = types.SimpleNamespace(
        names={0: "person", 1: "robot"},
        trainer=types.SimpleNamespace(best=str(best), save_dir=str(run)),
    )
    t._register_trained_model()

    written = yaml_lib.safe_load((run / "data.yaml").read_text())
    assert written["keypoint_schema"] == schema


def test_register_no_kpt_shape_yields_names_only_yaml(tmp_path, qapp):
    # Non-pose run: training yaml has no kpt_shape -> no pose keys leak in,
    # confirming no regression for plain detect/segment training.
    run = tmp_path / "models" / "yolo" / "custom" / "detect_proj"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"fake")

    train_yaml_path = tmp_path / "train_data3.yaml"
    train_yaml_path.write_text(yaml_lib.dump({"names": {0: "cell"}, "nc": 1}))

    t = _trainer(tmp_path, qapp)
    t.main_window = types.SimpleNamespace(keypoint_schemas={})
    t.yaml_path = str(train_yaml_path)
    t.model = types.SimpleNamespace(
        names={0: "cell"},
        trainer=types.SimpleNamespace(best=str(best), save_dir=str(run)),
    )
    t._register_trained_model()

    written = yaml_lib.safe_load((run / "data.yaml").read_text())
    assert "kpt_shape" not in written
    assert "flip_idx" not in written
    assert "keypoint_schema" not in written
    assert written["names"] == {0: "cell"}


# --- load_prediction_model: pose schema reconstruction (#35 PR-3) ----------

class _FakeYOLOModel:
    """Stand-in for ultralytics.YOLO(model_path) — only `.names` is read."""

    def __init__(self, model_path):
        self.names = {0: "person"}


def test_load_prediction_model_uses_rich_keypoint_schema(tmp_path, qapp, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLOModel)
    schema = {
        "names": ["nose", "left_eye", "right_eye"],
        "skeleton": [[0, 1], [0, 2]],
        "flip_idx": [0, 2, 1],
    }
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml_lib.dump({
        "names": {0: "person"},
        "kpt_shape": [3, 3],
        "flip_idx": [0, 2, 1],
        "keypoint_schema": schema,
    }))

    t = _trainer(tmp_path, qapp)
    ok, msg = t.load_prediction_model(str(tmp_path / "best.pt"), str(yaml_path))

    assert ok is True and msg is None
    assert t.prediction_keypoint_schema == schema


def test_load_prediction_model_falls_back_to_generic_names(tmp_path, qapp, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLOModel)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml_lib.dump({
        "names": {0: "person"},
        "kpt_shape": [3, 3],
        "flip_idx": [0, 2, 1],
    }))

    t = _trainer(tmp_path, qapp)
    ok, msg = t.load_prediction_model(str(tmp_path / "best.pt"), str(yaml_path))

    assert ok is True and msg is None
    assert t.prediction_keypoint_schema == {
        "names": ["kp0", "kp1", "kp2"],
        "skeleton": [],
        "flip_idx": [0, 2, 1],
    }


def test_load_prediction_model_none_schema_for_non_pose_yaml(tmp_path, qapp, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLOModel)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml_lib.dump({"names": {0: "person"}}))

    t = _trainer(tmp_path, qapp)
    ok, msg = t.load_prediction_model(str(tmp_path / "best.pt"), str(yaml_path))

    assert ok is True and msg is None
    assert t.prediction_keypoint_schema is None


def test_load_prediction_model_sets_loaded_model_path(tmp_path, qapp, monkeypatch):
    # loaded_model_path must track EVERY self.model assignment so train_model's
    # per-run reload trains the model the user actually loaded last, not a stale
    # earlier one (senior-review P1, #35 PR-3).
    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLOModel)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml_lib.dump({"names": {0: "person"}}))
    model_path = str(tmp_path / "best.pt")

    t = _trainer(tmp_path, qapp)
    t.load_prediction_model(model_path, str(yaml_path))

    assert t.loaded_model_path == model_path


# --- _prune_run_artifacts ---------------------------------------------------

def _make_full_run_dir(tmp_path):
    """A run dir shaped like Ultralytics' output: weights + diagnostics."""
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")
    data_yaml = run / "data.yaml"
    data_yaml.write_text("names: {0: cell}\n")
    for noise in ("results.csv", "args.yaml", "BoxF1_curve.png",
                  "confusion_matrix.png", "train_batch0.jpg", "val_batch0_pred.jpg"):
        (run / noise).write_text("x")
    return run, best, data_yaml


def test_prune_keeps_only_best_and_yaml_when_tracked(tmp_path, qapp):
    run, best, data_yaml = _make_full_run_dir(tmp_path)
    t = _trainer(tmp_path, qapp)
    t._mlflow_url_emitted = True  # MLflow has the diagnostics -> safe to prune
    t._prune_run_artifacts(run, best, data_yaml)

    survivors = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    assert survivors == {"weights/best.pt", "data.yaml"}
    assert best.exists() and data_yaml.exists()
    # weights/ survives (best.pt still in it); no empty dirs left behind
    assert (run / "weights").is_dir()


def test_prune_kept_run_still_discoverable(tmp_path, qapp, monkeypatch):
    # After pruning, list_custom_yolo_models() must still find the run.
    custom = tmp_path / "custom"
    run = custom / "proj"
    (run / "weights").mkdir(parents=True)
    best = run / "weights" / "best.pt"
    best.write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")
    data_yaml = run / "data.yaml"
    data_yaml.write_text("names: {0: cell}\n")
    (run / "results.csv").write_text("x")

    t = _trainer(tmp_path, qapp)
    t._mlflow_url_emitted = True
    t._prune_run_artifacts(run, best, data_yaml)

    monkeypatch.setattr(yt, "YOLO_CUSTOM_DIR", str(custom))
    listed = list_custom_yolo_models()
    assert set(listed) == {"★ proj"}
    (model_path, yaml_path), = listed.values()
    assert model_path.endswith("best.pt") and yaml_path.endswith("data.yaml")


def test_prune_keeps_everything_when_not_tracked(tmp_path, qapp):
    run, best, data_yaml = _make_full_run_dir(tmp_path)
    before = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    t = _trainer(tmp_path, qapp)
    t._mlflow_url_emitted = False  # diagnostics live nowhere else -> keep them
    t._prune_run_artifacts(run, best, data_yaml)

    after = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    assert after == before  # nothing deleted


# --- _emit_mlflow_url -------------------------------------------------------

def test_emit_mlflow_url_once(tmp_path, qapp, monkeypatch):
    run_info = types.SimpleNamespace(
        info=types.SimpleNamespace(experiment_id="7", run_id="abc123")
    )
    fake_mlflow = types.SimpleNamespace(active_run=lambda: run_info)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    t = _trainer(tmp_path, qapp)
    urls = []
    t.mlflow_run_url.connect(urls.append)

    t._emit_mlflow_url()
    t._emit_mlflow_url()  # latched — must not emit again

    assert urls == ["http://localhost:5000/#/experiments/7/runs/abc123"]


def test_emit_mlflow_url_silent_when_no_active_run(tmp_path, qapp, monkeypatch):
    fake_mlflow = types.SimpleNamespace(active_run=lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake_mlflow)

    t = _trainer(tmp_path, qapp)
    urls = []
    t.mlflow_run_url.connect(urls.append)
    t._emit_mlflow_url()
    assert urls == []
    assert t._mlflow_url_emitted is False  # not latched -> a later epoch retries


# --- LoadPredictionModelDialog dropdown -------------------------------------

def test_prediction_dialog_lists_and_fills_trained_models(tmp_path, qapp, qtbot, monkeypatch):
    mapping = {
        "★ runA": ("/models/runA/weights/best.pt", "/models/runA/data.yaml"),
    }
    monkeypatch.setattr(yt, "list_custom_yolo_models", lambda: mapping)

    dlg = LoadPredictionModelDialog()
    qtbot.addWidget(dlg)
    # placeholder + one trained entry
    assert dlg.trained_combo.count() == 2

    dlg.trained_combo.setCurrentIndex(1)  # select runA
    assert dlg.model_path == "/models/runA/weights/best.pt"
    assert dlg.yaml_path == "/models/runA/data.yaml"
    assert dlg.model_edit.text() == "/models/runA/weights/best.pt"
    assert dlg.yaml_edit.text() == "/models/runA/data.yaml"


def test_prediction_dialog_placeholder_is_noop(tmp_path, qapp, qtbot, monkeypatch):
    # Re-selecting the placeholder (currentData() is None) must NOT wipe a
    # previously chosen model — the `if not paths: return` guard is load-bearing.
    mapping = {
        "★ runA": ("/models/runA/weights/best.pt", "/models/runA/data.yaml"),
    }
    monkeypatch.setattr(yt, "list_custom_yolo_models", lambda: mapping)
    dlg = LoadPredictionModelDialog()
    qtbot.addWidget(dlg)

    dlg.trained_combo.setCurrentIndex(1)   # pick runA
    dlg.trained_combo.setCurrentIndex(0)   # back to "— select a trained model —"
    assert dlg.model_path == "/models/runA/weights/best.pt"
    assert dlg.yaml_path == "/models/runA/data.yaml"


def test_prediction_dialog_no_combo_when_no_trained_models(tmp_path, qapp, qtbot, monkeypatch):
    monkeypatch.setattr(yt, "list_custom_yolo_models", dict)
    dlg = LoadPredictionModelDialog()
    qtbot.addWidget(dlg)
    assert not hasattr(dlg, "trained_combo")  # dropdown omitted entirely
