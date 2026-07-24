"""ClassController unit tests (fork issue #35).

Class management mutates every annotation in the project — rename re-keys,
delete drops — yet `controllers/class_controller.py` had no tests. These
build one real ImageAnnotator (offscreen) and drive the controller directly.

`window.auto_save` is monkeypatched to a no-op in the fixture so the
add/rename/delete/colour paths (which all call it) can't open a save prompt.
No production code is changed — tests only.
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor


@pytest.fixture
def window(qt_application, monkeypatch):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    # Every mutating class op calls auto_save(); no-op it so no save prompt
    # (QFileDialog/QMessageBox) can appear in the offscreen run.
    monkeypatch.setattr(w, "auto_save", lambda *a, **k: None)
    yield w
    w.deleteLater()


@pytest.fixture(autouse=True)
def _no_native_dialogs(monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))


POSE = {"keypoints": [3.0, 3.0, 2], "num_keypoints": 1,
        "bbox": [2.0, 2.0, 4.0, 4.0], "category_id": 1,
        "category_name": "cell", "number": 1}


def _first_item(window):
    return window.class_list.item(0)


def test_add_class_programmatic(window):
    window.add_class("cell", QColor("#ff0000"))

    assert window.class_mapping["cell"] == 1
    assert window.image_label.class_colors["cell"].name() == "#ff0000"
    assert window.class_list.count() == 1
    assert window.current_class == "cell"


def test_add_duplicate_class_noop(window):
    window.add_class("cell", QColor("#ff0000"))
    window.add_class("cell", QColor("#00ff00"))  # duplicate — must be a no-op

    assert window.class_list.count() == 1
    assert window.class_mapping == {"cell": 1}


def test_rename_class_rekeys_everything(window, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    window.add_class("cell", QColor("#ff0000"))
    old_id = window.class_mapping["cell"]
    window.keypoint_schemas["cell"] = {"names": ["a"], "skeleton": [], "flip_idx": [0]}
    window.all_annotations["a.png"] = {
        "cell": [{"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0],
                  "category_name": "cell"}]
    }

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("blob", True))
    )
    window.class_controller.rename_class(_first_item(window))

    assert window.class_mapping["blob"] == old_id
    assert "cell" not in window.class_mapping
    assert "cell" not in window.image_label.class_colors
    assert "cell" not in window.keypoint_schemas
    assert "cell" not in window.all_annotations["a.png"]
    assert "blob" in window.all_annotations["a.png"]
    for ann in window.all_annotations["a.png"]["blob"]:
        assert ann["category_name"] == "blob"


def test_delete_class_removes_annotations(window):
    window.add_class("cell", QColor("#ff0000"))
    window.all_annotations["a.png"] = {
        "cell": [{"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0],
                  "category_name": "cell"}]
    }
    item = _first_item(window)

    window.class_controller.delete_class(item)

    assert "cell" not in window.class_mapping
    assert "cell" not in window.image_label.class_colors
    assert "cell" not in window.all_annotations["a.png"]
    assert window.class_list.count() == 0


def test_delete_class_accepts_string(window):
    """delete_selected_class passes the class *name* (a str), not the item."""
    window.add_class("cell", QColor("#ff0000"))
    window.all_annotations["a.png"] = {
        "cell": [{"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0],
                  "category_name": "cell"}]
    }

    window.class_controller.delete_class("cell")

    assert "cell" not in window.class_mapping
    assert "cell" not in window.all_annotations["a.png"]
    assert window.class_list.count() == 0


def test_change_class_color(window, monkeypatch):
    from PyQt6.QtWidgets import QColorDialog

    window.add_class("cell", QColor("#ff0000"))
    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#00ff00"))
    )

    window.class_controller.change_class_color(_first_item(window))

    assert window.image_label.class_colors["cell"].name() == "#00ff00"


def test_visibility_toggle(window):
    window.add_class("cell", QColor("#ff0000"))
    item = _first_item(window)

    item.setCheckState(Qt.CheckState.Unchecked)
    window.class_controller.toggle_class_visibility(item)
    assert window.image_label.class_visibility["cell"] is False
    assert window.class_controller.is_class_visible("cell") is False

    item.setCheckState(Qt.CheckState.Checked)
    window.class_controller.toggle_class_visibility(item)
    assert window.image_label.class_visibility["cell"] is True
    assert window.class_controller.is_class_visible("cell") is True


def test_class_has_keypoint_instances(window):
    window.add_class("cell", QColor("#ff0000"))
    ctrl = window.class_controller

    assert ctrl._class_has_keypoint_instances("cell") is False

    window.all_annotations["a.png"] = {"cell": [dict(POSE)]}
    assert ctrl._class_has_keypoint_instances("cell") is True
