"""Unit tests for DINOController's Temp-class review carrying keypoint
schemas over on accept/reject (issue #35 PR-3).

``accept_visible_temp_classes`` / ``reject_visible_temp_classes`` move a
reviewed ``"Temp-<class>"`` annotation bucket (produced by DINO detection or
YOLO-pose prediction review) into/out of the permanent class set. When the
temp bucket carries a per-class keypoint schema
(``mw.keypoint_schemas["Temp-<class>"]``), accept must carry it over to the
permanent class name -- or warn and keep the existing schema on a K
mismatch -- and reject must never leave an orphaned ``"Temp-<class>"``
schema entry behind.

Uses a lightweight main-window stand-in rather than a full ImageAnnotator:
a real ``QListWidget`` for ``class_list`` (DINOController relies on its Qt
wildcard ``findItems``/``count``/``item`` API), plain stubs for everything
else. The controller is built via ``__new__`` to skip ``QObject.__init__``
-- same pattern as ``tests/unit/test_yolo_training_args.py``'s
``_make_trainer`` (skip __init__, which needs a real main window; set just
what the methods under test touch).
"""

import types

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

from digitalsreeni_image_annotator.controllers import dino_controller as dc


class _MW:
    """Minimal main-window stand-in exposing exactly what
    accept/reject_visible_temp_classes touch."""

    def __init__(self):
        self.class_list = QListWidget()
        self.image_label = types.SimpleNamespace(
            annotations={}, class_colors={}, update=lambda: None,
        )
        self.keypoint_schemas = {}
        self.class_mapping = {}
        self.current_class = None
        self.current_slice = None
        self.image_file_name = "img.png"
        self.all_annotations = {}
        self.add_class_calls = []
        self.on_class_selected_calls = []

    def add_class(self, name, color=None):
        # Mirrors the essential effect of class_controller.add_class for
        # this test's purposes (schema handling doesn't depend on the rest).
        self.add_class_calls.append(name)
        self.class_mapping.setdefault(name, len(self.class_mapping) + 1)
        self.image_label.class_colors.setdefault(name, color)

    def update_class_list(self):
        pass

    def update_annotation_list(self):
        pass

    def save_current_annotations(self):
        pass

    def on_class_selected(self, item=None):
        self.on_class_selected_calls.append(item)

    def disable_annotation_tools(self):
        pass


def _list_item(name, checked=True):
    item = QListWidgetItem(name)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


@pytest.fixture
def mw(qt_application):
    return _MW()


@pytest.fixture
def controller(mw):
    c = dc.DINOController.__new__(dc.DINOController)
    c.mw = mw
    return c


def _temp_pose_annotation(category="Temp-X"):
    return {
        "keypoints": [1, 2, 2, 3, 4, 2],
        "num_keypoints": 2,
        "bbox": [0, 0, 5, 5],
        "category_name": category,
        "score": 0.9,
        "temp": True,
    }


_SCHEMA_K2 = {"names": ["a", "b"], "skeleton": [[0, 1]], "flip_idx": [0, 1]}
_SCHEMA_K3 = {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": [0, 1, 2]}


def _checked_temp_item(mw, name="Temp-X"):
    mw.class_list.addItem(_list_item(name, checked=True))


# --- accept: schema carry-over ----------------------------------------------

def test_accept_new_permanent_class_carries_schema(mw, controller, monkeypatch):
    monkeypatch.setattr(dc.QMessageBox, "information", lambda *a, **k: None)
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    mw.keypoint_schemas["Temp-X"] = dict(_SCHEMA_K2)
    _checked_temp_item(mw)

    controller.accept_visible_temp_classes()

    assert mw.keypoint_schemas["X"] == _SCHEMA_K2
    assert "Temp-X" not in mw.keypoint_schemas


def test_accept_into_existing_same_k_keeps_existing(mw, controller, monkeypatch):
    monkeypatch.setattr(dc.QMessageBox, "information", lambda *a, **k: None)
    existing = dict(_SCHEMA_K2)
    mw.image_label.annotations["X"] = []
    mw.keypoint_schemas["X"] = existing
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    # Same K as the existing schema but different names -- accept must keep
    # the existing (hand-authored) schema rather than overwrite it.
    mw.keypoint_schemas["Temp-X"] = {"names": ["p", "q"], "skeleton": [], "flip_idx": [0, 1]}
    _checked_temp_item(mw)

    controller.accept_visible_temp_classes()

    assert mw.keypoint_schemas["X"] == existing
    assert mw.keypoint_schemas["X"] is existing
    assert "Temp-X" not in mw.keypoint_schemas


def test_accept_into_existing_different_k_warns_and_keeps_existing(mw, controller, monkeypatch):
    monkeypatch.setattr(dc.QMessageBox, "information", lambda *a, **k: None)
    warnings = []
    monkeypatch.setattr(dc.QMessageBox, "warning", lambda *a, **k: warnings.append(a))
    existing = dict(_SCHEMA_K3)
    mw.image_label.annotations["X"] = []
    mw.keypoint_schemas["X"] = existing
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    mw.keypoint_schemas["Temp-X"] = dict(_SCHEMA_K2)  # K=2 vs existing K=3
    _checked_temp_item(mw)

    controller.accept_visible_temp_classes()

    assert warnings, "expected a QMessageBox.warning for the schema K mismatch"
    assert mw.keypoint_schemas["X"] == existing  # kept, not overwritten by the temp schema
    assert "Temp-X" not in mw.keypoint_schemas


# --- reject: orphaned schema cleanup ----------------------------------------

def test_reject_removes_orphaned_schema(mw, controller):
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    mw.keypoint_schemas["Temp-X"] = dict(_SCHEMA_K2)
    _checked_temp_item(mw)

    controller.reject_visible_temp_classes()

    assert "Temp-X" not in mw.keypoint_schemas


# --- no regression: no keypoint schema involved at all ----------------------

def test_accept_without_schema_entry_no_crash_no_mutation(mw, controller, monkeypatch):
    monkeypatch.setattr(dc.QMessageBox, "information", lambda *a, **k: None)
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    _checked_temp_item(mw)

    controller.accept_visible_temp_classes()  # must not raise KeyError

    assert mw.keypoint_schemas == {}  # untouched
    assert mw.image_label.annotations["X"][0]["category_name"] == "X"
    assert "Temp-X" not in mw.image_label.annotations


def test_reject_without_schema_entry_no_crash_no_mutation(mw, controller):
    mw.image_label.annotations["Temp-X"] = [_temp_pose_annotation()]
    mw.image_label.class_colors["Temp-X"] = "red"
    _checked_temp_item(mw)

    controller.reject_visible_temp_classes()  # must not raise KeyError

    assert mw.keypoint_schemas == {}
    assert "Temp-X" not in mw.image_label.annotations
