"""
Unit tests for the YOLO trainer forwarding the LR-schedule + early-stop knobs
to Ultralytics' ``model.train`` (issue #85).

`train_model` must pass through ``cos_lr`` / ``lr0`` / ``lrf`` / ``warmup_epochs``
/ ``patience``, deriving ``warmup_epochs`` as ~10% of epochs when not given. The
real ``model.train`` is replaced by a recorder so no training actually runs.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from src.digitalsreeni_image_annotator.controllers.yolo_controller import (
    build_yolo_train_opts,
)
from src.digitalsreeni_image_annotator.dialogs import yolo_trainer as yt


def test_train_opts_on_is_warmup_cosine():
    opts = build_yolo_train_opts(50, cos_lr=True, lr0=0.01, patience=20)
    assert opts == {
        "cos_lr": True, "lr0": 0.01, "lrf": 0.1, "warmup_epochs": 5, "patience": 20,
    }


def test_train_opts_off_is_constant_lr():
    # cos_lr=False + lrf=1.0 (no decay) + warmup_epochs=0 ⇒ genuinely constant LR,
    # matching the SAM schedule toggle's "off" semantics.
    opts = build_yolo_train_opts(50, cos_lr=False, lr0=0.02, patience=0)
    assert opts["cos_lr"] is False
    assert opts["lrf"] == 1.0
    assert opts["warmup_epochs"] == 0
    assert opts["lr0"] == 0.02 and opts["patience"] == 0


class _FakeModel:
    def __init__(self):
        self.train_kwargs = None
        self.callbacks = {"on_train_epoch_end": [], "on_fit_epoch_end": []}

    def add_callback(self, name, cb):
        self.callbacks.setdefault(name, []).append(cb)

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return "results"


# Default (non-pose) dataset yaml used by most _make_trainer callers.
_PLAIN_YAML = "train: images/train\nval: images/train\nnc: 1\nnames: [a]\n"

# Pose dataset yaml — the only extra field train_model's pre-flight guard
# checks for is the presence of "kpt_shape".
_POSE_YAML = (
    "train: images/train\nval: images/train\nnc: 1\nnames: [a]\n"
    "kpt_shape: [2, 3]\nflip_idx: [0, 1]\n"
)


def _make_trainer(tmp_path, monkeypatch, yaml_text=None):
    (tmp_path / "images" / "train").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(_PLAIN_YAML if yaml_text is None else yaml_text, encoding="utf-8")

    # Skip __init__ (needs a real main window); set just what train_model touches.
    trainer = yt.YOLOTrainer.__new__(yt.YOLOTrainer)
    trainer.project_dir = str(tmp_path)
    trainer.yaml_path = str(yaml_path)
    trainer.stop_training = False
    trainer.epoch_info = deque(maxlen=10)
    trainer._mlflow_url_emitted = False
    trainer.loaded_model_path = None  # no reload by default (see the reload test)
    trainer.model = _FakeModel()

    monkeypatch.setattr(trainer, "_register_trained_model", lambda: None)
    # train_model resolves the device via ..core.torch_utils; pin it to CPU.
    from src.digitalsreeni_image_annotator.core import torch_utils
    monkeypatch.setattr(torch_utils, "resolve_torch_device", lambda: ("cpu", None))
    return trainer


def test_forwards_schedule_and_early_stop_args(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=50, imgsz=320, cos_lr=True, lr0=0.02, lrf=0.1, patience=15)

    kw = trainer.model.train_kwargs
    assert kw["epochs"] == 50 and kw["imgsz"] == 320
    assert kw["cos_lr"] is True
    assert kw["lr0"] == 0.02
    assert kw["lrf"] == 0.1
    assert kw["patience"] == 15
    assert kw["warmup_epochs"] == 5  # round(0.1 * 50)


@pytest.mark.parametrize("epochs,expected", [(50, 5), (3, 1), (10, 1)])
def test_warmup_epochs_defaults_to_ten_percent(tmp_path, monkeypatch, epochs, expected):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=epochs, imgsz=640)  # warmup_epochs not given
    assert trainer.model.train_kwargs["warmup_epochs"] == expected


def test_registers_both_progress_callbacks(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=5, imgsz=640)
    # Cleared in the finally, so they end empty — but both keys must exist
    # (proving on_fit_epoch_end is wired alongside on_train_epoch_end).
    assert "on_train_epoch_end" in trainer.model.callbacks
    assert "on_fit_epoch_end" in trainer.model.callbacks
    assert trainer.model.callbacks["on_fit_epoch_end"] == []


# ── on_fit_epoch_end metric surfacing (val loss + mAP + lr) ──────────────────

def _fit_trainer(metrics, lr={"lr/pg0": 0.0012}):
    return SimpleNamespace(epoch=2, epochs=10, metrics=metrics, lr=lr)


def _emit_fit_line(tmp_path, metrics, lr={"lr/pg0": 0.0012}):
    trainer = yt.YOLOTrainer(str(tmp_path), None)
    out = []
    trainer.progress_signal.connect(out.append)
    trainer.on_fit_epoch_end(_fit_trainer(metrics, lr))
    return out[-1] if out else ""


def test_fit_line_parenthesized_map_keys(qtbot, tmp_path):
    # Ultralytics' in-memory trainer.metrics uses "(B)"-suffixed mAP keys.
    line = _emit_fit_line(tmp_path, {
        "val/box_loss": 0.59532,
        "metrics/mAP50(B)": 0.34212,
        "metrics/mAP50-95(B)": 0.28817,
    })
    assert "Epoch 3/10" in line
    assert "val_box_loss=0.5953" in line
    assert "mAP50=0.3421" in line and "mAP50-95=0.2882" in line
    assert "lr=1.20e-03" in line


def test_fit_line_paren_stripped_map_keys(qtbot, tmp_path):
    # Real MLflow-store form (parens stripped) — must still be surfaced.
    line = _emit_fit_line(tmp_path, {
        "val/box_loss": 0.59532,
        "metrics/mAP50B": 0.34212,
        "metrics/mAP50-95B": 0.28817,
    })
    assert "mAP50=0.3421" in line and "mAP50-95=0.2882" in line


def test_fit_line_segmentation_surfaces_seg_loss(qtbot, tmp_path):
    line = _emit_fit_line(tmp_path, {
        "val/box_loss": 0.5, "val/seg_loss": 0.31, "metrics/mAP50(M)": 0.4,
    })
    assert "val_seg_loss=0.3100" in line and "mAP50=0.4000" in line


def test_fit_line_no_crash_on_empty_metrics(qtbot, tmp_path):
    # Defensive: missing metrics / lr must never raise; nothing meaningful to show.
    line = _emit_fit_line(tmp_path, {}, lr=None)
    assert line == ""  # only the epoch tag would remain → not emitted


# ── prepare_dataset() forwards keypoint_schemas (#35 PR-3) ───────────────────

def test_prepare_dataset_passes_keypoint_schemas(tmp_path, monkeypatch):
    trainer = yt.YOLOTrainer.__new__(yt.YOLOTrainer)
    trainer.dataset_path = str(tmp_path / "yolo_dataset")

    schemas = {"person": {"names": ["nose", "eye"], "skeleton": [[0, 1]], "flip_idx": [0, 1]}}
    trainer.main_window = SimpleNamespace(
        all_annotations={}, class_mapping={}, image_paths={}, slices={},
        image_slices={}, keypoint_schemas=schemas,
    )

    # export_yolo_v5plus itself is monkeypatched to a recorder — prepare_dataset
    # still needs a real yaml file at the returned path since it reads/rewrites it.
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(_PLAIN_YAML, encoding="utf-8")
    calls = {}

    def _fake_export(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return str(tmp_path), str(yaml_path)

    monkeypatch.setattr(yt, "export_yolo_v5plus", _fake_export)

    trainer.prepare_dataset(val_split=20)

    assert calls["kwargs"]["keypoint_schemas"] is schemas


# ── train_model() pose/non-pose pre-flight guard (#35 PR-3) ──────────────────

def test_train_model_guard_pose_model_needs_pose_yaml(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, yaml_text=_PLAIN_YAML)
    trainer.model.task = "pose"

    with pytest.raises(ValueError, match="pose"):
        trainer.train_model(epochs=5, imgsz=640)
    assert trainer.model.train_kwargs is None  # never reached model.train()


def test_train_model_guard_pose_yaml_needs_pose_model(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, yaml_text=_POSE_YAML)
    trainer.model.task = "segment"

    with pytest.raises(ValueError):
        trainer.train_model(epochs=5, imgsz=640)
    assert trainer.model.train_kwargs is None  # never reached model.train()


def test_train_model_pose_model_with_pose_yaml_proceeds(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch, yaml_text=_POSE_YAML)
    trainer.model.task = "pose"

    trainer.train_model(epochs=5, imgsz=640)

    assert trainer.model.train_kwargs is not None
    assert trainer.model.train_kwargs["epochs"] == 5


def test_train_model_non_pose_model_with_non_pose_yaml_proceeds(tmp_path, monkeypatch):
    # Regression check: non-pose runs (the pre-existing behavior) must be unaffected.
    trainer = _make_trainer(tmp_path, monkeypatch, yaml_text=_PLAIN_YAML)
    trainer.model.task = "segment"

    trainer.train_model(epochs=5, imgsz=640)

    assert trainer.model.train_kwargs is not None
    assert trainer.model.train_kwargs["epochs"] == 5


# ── on_fit_epoch_end() pose metrics (#35 PR-3) ────────────────────────────────

def test_fit_line_pose_surfaces_pose_and_kobj_loss(qtbot, tmp_path):
    line = _emit_fit_line(tmp_path, {
        "val/box_loss": 0.5, "val/pose_loss": 0.22, "val/kobj_loss": 0.11,
    })
    assert "val_pose_loss=0.2200" in line and "val_kobj_loss=0.1100" in line


def test_fit_line_pose_map_key_paren_suffixed(qtbot, tmp_path):
    # No source change backs this test — it confirms the EXISTING substring
    # mAP matcher (already task-suffix-agnostic for "(B)"/"(M)") also handles
    # pose's "(P)" suffix without any code change.
    line = _emit_fit_line(tmp_path, {
        "val/box_loss": 0.5, "metrics/mAP50(P)": 0.6123, "metrics/mAP50-95(P)": 0.4321,
    })
    assert "mAP50=0.6123" in line and "mAP50-95=0.4321" in line


# ── consecutive-run reload (#35 PR-3 manual-testing fix) ─────────────────────

def test_train_model_reloads_pristine_model_each_run(tmp_path, monkeypatch):
    """Ultralytics drops overrides['model'] during train(), so a second train()
    on the same YOLO object raises KeyError('model'). train_model reloads a
    pristine model from loaded_model_path before every run so consecutive
    trainings work."""
    import ultralytics

    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.loaded_model_path = "pretrained-pose.pt"

    reloaded = []

    def _fake_yolo(path):
        reloaded.append(path)
        return _FakeModel()

    # train_model does `from ultralytics import YOLO` at call time, so patching
    # the ultralytics module attribute takes effect.
    monkeypatch.setattr(ultralytics, "YOLO", _fake_yolo)

    trainer.train_model(epochs=3, imgsz=640)
    first_model = trainer.model
    trainer.train_model(epochs=3, imgsz=640)

    # Reloaded from the pristine checkpoint before BOTH runs, and each run got a
    # fresh object (the second run is NOT the first run's mutated instance).
    assert reloaded == ["pretrained-pose.pt", "pretrained-pose.pt"]
    assert trainer.model is not first_model
    assert trainer.model.train_kwargs is not None


def test_train_model_no_reload_when_path_unset(tmp_path, monkeypatch):
    """Defensive: if no load path was recorded, keep the existing model object
    (no crash, no reload attempt)."""
    import ultralytics

    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.loaded_model_path = None
    original = trainer.model
    monkeypatch.setattr(ultralytics, "YOLO", lambda *a, **k: pytest.fail("must not reload"))

    trainer.train_model(epochs=3, imgsz=640)

    assert trainer.model is original


def test_train_model_guard_runs_against_reloaded_model(tmp_path, monkeypatch):
    """The pre-flight task guard must validate the RELOADED model (reload runs
    first). A pose yaml + a reloaded NON-pose model must raise, even though the
    pre-reload model claimed pose — proving the guard sees the reloaded one."""
    import ultralytics

    trainer = _make_trainer(tmp_path, monkeypatch, yaml_text=_POSE_YAML)
    trainer.loaded_model_path = "some-detect-model.pt"
    trainer.model.task = "pose"  # pre-reload model would PASS the guard

    def _fake_yolo(path):
        m = _FakeModel()
        m.task = "segment"  # reloaded model must FAIL the pose-yaml guard
        return m

    monkeypatch.setattr(ultralytics, "YOLO", _fake_yolo)

    with pytest.raises(ValueError):
        trainer.train_model(epochs=5, imgsz=640)
    assert trainer.model.train_kwargs is None  # never reached model.train()


# ── predict() no longer hardcodes task= (#35 PR-3) ────────────────────────────

def test_predict_does_not_pass_task_kwarg(tmp_path, monkeypatch):
    from src.digitalsreeni_image_annotator.core import torch_utils
    monkeypatch.setattr(torch_utils, "resolve_torch_device", lambda: ("cpu", None))

    trainer = yt.YOLOTrainer.__new__(yt.YOLOTrainer)
    trainer.conf_threshold = 0.25

    calls = {}

    class _RecordingModel:
        def __call__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            fake_result = SimpleNamespace(
                orig_shape=(100, 100),
                orig_img=np.zeros((100, 100, 3), dtype=np.uint8),
            )
            return [fake_result]

    trainer.model = _RecordingModel()

    results, input_size, original_size = trainer.predict("fake_input.jpg")

    assert "task" not in calls["kwargs"]
    assert input_size == (100, 100)
    assert original_size == (100, 100)
