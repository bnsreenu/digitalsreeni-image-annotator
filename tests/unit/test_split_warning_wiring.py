"""The split warning's three call sites (issue #81, ADR-044).

``split_warning`` itself is pure text and tested in ``test_dataset_split.py``.
This file covers the part that decides whether anyone ever sees it — which is
the load-bearing half, and which had no coverage at all until a review pointed
out that the exporter's wiring got a test and the dialogs' did not.

The warning offers **Cancel**, and every caller honours it. A warning saying
the validation numbers cannot be trusted, with only an OK button, trains
exactly the click-through reflex it exists to prevent.
"""

import pytest
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from src.digitalsreeni_image_annotator.controllers import io_controller


def _frames(base, count):
    return [f"{base}_F{i:05d}" for i in range(count)]


class _Recorder:
    """Stands in for QMessageBox.warning, recording what it was shown."""

    def __init__(self, answer):
        self.answer = answer
        self.messages = []

    def __call__(self, _parent, _title, message, *_args, **_kwargs):
        self.messages.append(message)
        return self.answer


@pytest.fixture
def warning_box(monkeypatch):
    def _install(answer=QMessageBox.StandardButton.Ok):
        recorder = _Recorder(answer)
        monkeypatch.setattr(QMessageBox, "warning", recorder)
        return recorder

    return _install


# --- confirm_split_warning --------------------------------------------------


def test_a_healthy_split_shows_nothing_and_proceeds(warning_box):
    box = warning_box()
    names = _frames("a", 10) + _frames("b", 10)
    assert io_controller.confirm_split_warning(None, names, None, 20) is True
    assert box.messages == []


def test_a_degenerate_split_is_shown_and_can_be_accepted(warning_box):
    box = warning_box(QMessageBox.StandardButton.Ok)
    assert io_controller.confirm_split_warning(None, _frames("a", 10), None, 20) is True
    assert len(box.messages) == 1
    assert "optimistic" in box.messages[0]


def test_declining_the_warning_backs_out(warning_box):
    warning_box(QMessageBox.StandardButton.Cancel)
    assert io_controller.confirm_split_warning(None, _frames("a", 10), None, 20) is False


# --- prompt_validation_split ------------------------------------------------


def test_the_prompt_returns_the_chosen_percentage(monkeypatch, warning_box):
    warning_box()
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (30, True))
    names = _frames("a", 10) + _frames("b", 10)
    assert io_controller.prompt_validation_split(None, names, None) == (30, True)


def test_cancelling_the_prompt_reports_not_ok(monkeypatch, warning_box):
    warning_box()
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (20, False))
    _value, ok = io_controller.prompt_validation_split(None, _frames("a", 10), None)
    assert ok is False


def test_declining_the_warning_returns_to_the_prompt(monkeypatch, warning_box):
    """The advice is "choose a different percentage", so the dialog has to come
    back — otherwise it is advice with nothing to act on."""
    warning_box(QMessageBox.StandardButton.Cancel)
    answers = iter([(20, True), (0, True)])
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: next(answers))

    # 20% on a single recording warns and is declined; 0% asks for no val set
    # at all, so there is nothing to warn about and it goes through.
    assert io_controller.prompt_validation_split(
        None, _frames("a", 10), None
    ) == (0, True)


def test_a_caller_without_project_state_gets_the_plain_prompt(monkeypatch):
    """``names=None`` keeps the historical behaviour for callers with nothing
    to check — and must not touch QMessageBox at all."""
    def _explode(*_args, **_kwargs):
        raise AssertionError("no warning should be computed without names")

    monkeypatch.setattr(QMessageBox, "warning", _explode)
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (20, True))
    assert io_controller.prompt_validation_split(None) == (20, True)


# --- split_inputs: one grouping, computed once ------------------------------


def test_split_inputs_carries_the_curation_refinement(qtbot):
    """The one route by which near-duplicate clusters reach a split (ADR-045).

    These two files are independent by name — nothing structural links them —
    so if `split_inputs` returned the bare derived grouping (or None), the
    refinement would be silently inert and every split would simply look
    reasonable.
    """
    import numpy as np
    from PyQt6.QtWidgets import QListWidget, QWidget

    from src.digitalsreeni_image_annotator.controllers.curation_controller import (
        CurationController,
    )

    class _Window(QWidget):
        def __init__(self):
            super().__init__()
            self.all_images = []
            self.image_paths = {
                name: f"C:/photos/{name}"
                for name in ("burst_a.png", "burst_b.png", "other.png")
            }
            self.image_slices = {}
            self.slices = []
            self.image_list = QListWidget()
            self.current_project_file = None
            self.all_annotations = {
                name: {"cell": [{"bbox": [0, 0, 1, 1]}]} for name in self.image_paths
            }

    window = _Window()
    qtbot.addWidget(window)
    window.curation_controller = CurationController(window)
    window.curation_controller.embeddings = {
        "burst_a.png": np.array([1.0, 0.0], dtype=np.float32),
        "burst_b.png": np.array([1.0, 0.01], dtype=np.float32),
        "other.png": np.array([0.0, 1.0], dtype=np.float32),
    }

    names, groups = io_controller.split_inputs(window)

    assert set(names) == set(window.all_annotations)
    assert groups is not None, "no grouping reached the split"
    assert groups["burst_a.png"] == groups["burst_b.png"]
    assert groups["other.png"] != groups["burst_a.png"]


# --- the training call sites ------------------------------------------------


class _FakeTrainer:
    def __init__(self):
        self.prepared = False

    def load_model(self, _base):
        return True

    def prepare_dataset(self, _val_split, groups=None):
        self.prepared = True
        self.groups = groups
        return "unused.yaml"


class _NoCurationRun:
    """A main window whose curation controller holds no embeddings.

    Delegates to the same ``derive_groups`` the controller would, rather than
    reimplementing it: the point of the stub is "no clusters to fold in", not
    "a different grouping".
    """

    def __init__(self, image_slices):
        self.image_slices = image_slices

    def split_groups(self, names):
        from src.digitalsreeni_image_annotator.core.dataset_split import (
            derive_groups,
        )

        return derive_groups(names, self.image_slices)


def test_declining_the_warning_abandons_a_yolo_run(warning_box, monkeypatch):
    """The unified Train dialog has its own split slider and never passes
    through the prompt, so this is its only chance to hear about the split."""
    from src.digitalsreeni_image_annotator.controllers.training_controller import (
        TrainingController,
    )

    warning_box(QMessageBox.StandardButton.Cancel)
    trainer = _FakeTrainer()

    class _MainWindow:
        all_annotations = {
            name: {"cell": [{"bbox": [0, 0, 1, 1]}]} for name in _frames("clip", 8)
        }
        image_paths = {}
        slices = [(name, object()) for name in _frames("clip", 8)]
        image_slices = {"clip": slices}
        yolo_trainer = trainer
        curation_controller = _NoCurationRun(image_slices)

        class yolo_controller:
            @staticmethod
            def initialize_yolo_trainer():
                pass

            @staticmethod
            def start_training(*_args):
                raise AssertionError("training must not start")

    controller = TrainingController.__new__(TrainingController)
    controller.mw = _MainWindow()
    controller.run_yolo({"base_model": "yolo11n-seg.pt", "val_split": 20})

    assert not trainer.prepared, "the dataset was prepared despite the warning"
