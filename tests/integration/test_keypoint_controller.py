"""Keypoint/pose annotation ↔ controller/persistence integration tests (#35).

One real offscreen ImageAnnotator. Covers finish_keypoint (placement → stored
instance + list row), single-point drag commit + undo, the merge and
change-class guards, delete+undo, and the per-class schema's `.iap` round-trip
(save → load) plus rename/delete propagation.
"""

import copy
import json

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


@pytest.fixture
def window(qt_application, monkeypatch):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    monkeypatch.setattr(w, "auto_save", lambda: None)
    yield w
    w.deleteLater()


class _FakeEvent:
    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


_SCHEMA = {"names": ["nose", "l_eye", "r_eye"], "skeleton": [[0, 1], [0, 2]], "flip_idx": [0, 2, 1]}


def _pose_class(window, name="person", schema=None):
    window.add_class(name)
    window.keypoint_schemas[name] = copy.deepcopy(schema or _SCHEMA)
    window.current_class = name


def _ready_canvas(window, image="img.png"):
    window.image_file_name = image
    window.current_slice = None
    window.all_annotations.setdefault(image, {})
    # Deep-copy, mirroring the real load_image_annotations (annotation_controller.py)
    # — image_label.annotations and all_annotations[image] must be genuinely
    # distinct objects, or a copy-vs-live selection bug (list-widget UserRole
    # copies vs the live dict) can't surface in these tests.
    window.image_label.annotations = copy.deepcopy(window.all_annotations[image])
    window.current_image = QImage(100, 100, QImage.Format.Format_RGB32)
    il = window.image_label
    il.original_pixmap = QPixmap(100, 100)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    return il


def _place(window, points):
    il = window.image_label
    il.current_keypoints = list(points)
    il.drawing_keypoints = True
    il.keypoint_next_index = len(points)
    window.annotation_controller.finish_keypoint()


def _selected_data(window):
    # The annotations widget is a QTableWidget; selection is per-row, with the
    # annotation dict in column 0's UserRole. Dedupe selected cells to rows.
    tbl = window.annotation_list
    rows = sorted({idx.row() for idx in tbl.selectedIndexes()})
    return [tbl.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]


# --- placement --------------------------------------------------------------

def test_finish_keypoint_stores_instance_and_row(window):
    _pose_class(window)
    il = _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 1)])

    anns = window.all_annotations["img.png"]["person"]
    assert len(anns) == 1
    ann = anns[0]
    assert ann["keypoints"] == [10, 20, 2, 30, 20, 2, 50, 20, 1]
    assert ann["num_keypoints"] == 3
    assert ann["number"] == 1
    # bbox surrounds the labelled points (with margin), clamped to the image.
    x, y, w, h = ann["bbox"]
    assert 0 <= x and 0 <= y and x + w <= 100 and y + h <= 100
    assert window.annotation_list.rowCount() == 1
    # in-progress state was cleared
    assert il.current_keypoints == [] and not il.drawing_keypoints


def test_finish_early_pads_unplaced_points_with_v0(window):
    _pose_class(window)
    _ready_canvas(window)
    _place(window, [(10, 20, 2)])  # only 1 of 3 placed → Enter/finish-early

    ann = window.all_annotations["img.png"]["person"][0]
    assert len(ann["keypoints"]) == 9          # padded to K=3
    assert ann["keypoints"][3:] == [0, 0, 0, 0, 0, 0]
    assert ann["num_keypoints"] == 1


# --- single-point editing ---------------------------------------------------

def test_keypoint_drag_persists_and_is_undoable(window):
    _pose_class(window)
    il = _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    (live,) = window.all_annotations["img.png"]["person"]
    window.annotation_controller.apply_canvas_selection([live], "replace")

    # Grab point 0 at (10,20) and drag it to (15,25).
    assert il._keypoint_at(live, (10, 20)) == 0
    il._begin_keypoint_edit(il._live_annotation(live), 0, (10, 20))
    il._update_keypoint_drag((15, 25))
    il._commit_keypoint_drag((15, 25), _FakeEvent())

    saved = window.all_annotations["img.png"]["person"][0]["keypoints"]
    assert saved[:3] == [15, 25, 2]

    window.annotation_controller.undo()
    assert window.all_annotations["img.png"]["person"][0]["keypoints"][:3] == [10, 20, 2]


def test_right_click_toggle_visibility(window):
    _pose_class(window)
    il = _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    (live,) = window.all_annotations["img.png"]["person"]
    window.annotation_controller.apply_canvas_selection([live], "replace")

    il._toggle_keypoint_visibility(il._live_annotation(live), 0)
    assert window.all_annotations["img.png"]["person"][0]["keypoints"][2] == 1  # 2 -> 1
    il._toggle_keypoint_visibility(il._live_annotation(live), 0)
    assert window.all_annotations["img.png"]["person"][0]["keypoints"][2] == 2  # 1 -> 2


def test_list_selected_toggle_visibility_persists_and_stays_selected(window):
    """Selecting via the annotation LIST (not the canvas) puts a value-equal
    *copy* in highlighted_annotations (PyQt round-trips UserRole dicts as
    copies). The toggle must still mutate the live object AND re-point
    highlighted_annotations at it before committing — otherwise the post-commit
    re-mirror can't value-match the (now-changed) stale copy, and the row
    silently deselects instead of staying selected."""
    _pose_class(window)
    il = _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    (live,) = window.all_annotations["img.png"]["person"]

    window.annotation_list.selectRow(0)
    window.annotation_controller.update_highlighted_annotations()
    assert il.highlighted_annotations and il.highlighted_annotations[0] is not live

    shape = il._single_selected_shape()      # the list copy (geometry)
    live_obj = il._live_annotation(shape)    # press handler resolves to live
    assert live_obj is live

    il._toggle_keypoint_visibility(live_obj, 0)  # emits keypointEditCommitted

    assert window.all_annotations["img.png"]["person"][0]["keypoints"][2] == 1
    assert _selected_data(window) == [live]  # still selected after the rebuild


# --- guards -----------------------------------------------------------------

def test_merge_blocks_keypoint_instances(window, monkeypatch):
    _pose_class(window)
    _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    _place(window, [(11, 21, 2), (31, 21, 2), (51, 21, 2)])
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.annotation_controller.QMessageBox.warning",
        lambda *a, **k: warned.append(a[2]),
    )
    window.annotation_list.selectAll()
    window.annotation_controller.merge_annotations()
    assert warned and "merge" in warned[0].lower()
    # Both instances still present (none silently deleted).
    assert len(window.all_annotations["img.png"]["person"]) == 2


def _force_change_class(window, monkeypatch, target):
    """Drive change_annotation_class as if the user picked `target` in its
    dialog (which is built inline), returning the list of warning messages."""
    import digitalsreeni_image_annotator.controllers.annotation_controller as acmod
    monkeypatch.setattr(acmod.QDialog, "exec", lambda self: acmod.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(acmod.QComboBox, "currentText", lambda self: target)
    warned = []
    monkeypatch.setattr(acmod.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    monkeypatch.setattr(acmod.QMessageBox, "information", lambda *a, **k: None)
    window.annotation_controller.change_annotation_class()
    return warned


def test_change_class_blocks_keypoint_to_non_pose(window, monkeypatch):
    _pose_class(window)
    window.add_class("cell")  # normal (non-pose) class
    window.current_class = "person"
    _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    window.annotation_list.selectRow(0)

    warned = _force_change_class(window, monkeypatch, "cell")
    assert warned  # blocked with an Incompatible Class warning
    assert len(window.all_annotations["img.png"]["person"]) == 1  # unmoved


def test_change_class_blocks_normal_into_pose(window, monkeypatch):
    _pose_class(window)
    window.add_class("cell")
    _ready_canvas(window)
    # A plain bbox annotation in the normal class.
    window.all_annotations["img.png"]["cell"] = [
        {"bbox": [1, 1, 5, 5], "category_name": "cell", "category_id": window.class_mapping["cell"], "number": 1}
    ]
    window.image_label.annotations = window.all_annotations["img.png"]
    window.update_annotation_list()
    # select the cell row
    for r in range(window.annotation_list.rowCount()):
        if window.annotation_list.item(r, 0).data(Qt.ItemDataRole.UserRole).get("category_name") == "cell":
            window.annotation_list.selectRow(r)
            break

    warned = _force_change_class(window, monkeypatch, "person")
    assert warned
    assert "cell" in window.all_annotations["img.png"]  # unmoved


def test_iap_roundtrips_a_keypoint_instance(window, tmp_path, monkeypatch):
    # The image file isn't written to disk, so the reload would pop a "missing
    # images" modal — no-op it (keeps the in-memory annotations for the assert).
    monkeypatch.setattr(window.project_controller, "handle_missing_images", lambda *a, **k: None)
    _pose_class(window)
    _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 1)])
    window.all_images = [
        {"file_name": "img.png", "width": 100, "height": 100, "is_multi_slice": False}
    ]
    window.image_paths = {}
    window.current_project_file = str(tmp_path / "proj.iap")
    window.project_controller.save_project(show_message=False)

    data = json.loads((tmp_path / "proj.iap").read_text(encoding="utf-8"))
    saved = data["images"][0]["annotations"]["person"][0]
    assert saved["keypoints"] == [10, 20, 2, 30, 20, 2, 50, 20, 1]
    assert saved["num_keypoints"] == 3

    # Reload restores the instance verbatim.
    window.all_annotations.clear()
    window.keypoint_schemas.clear()
    window.project_controller.load_project_data(data)
    restored = window.all_annotations["img.png"]["person"][0]
    assert restored["keypoints"] == [10, 20, 2, 30, 20, 2, 50, 20, 1]


def test_delete_then_undo_restores_instance(window):
    _pose_class(window)
    _ready_canvas(window)
    _place(window, [(10, 20, 2), (30, 20, 2), (50, 20, 2)])
    (live,) = window.all_annotations["img.png"]["person"]
    window.annotation_controller.apply_canvas_selection([live], "replace")
    window.annotation_list.selectRow(0)

    window.annotation_controller.delete_selected_annotations()
    assert window.all_annotations["img.png"].get("person", []) == []
    window.annotation_controller.undo()
    assert len(window.all_annotations["img.png"]["person"]) == 1


# --- schema persistence -----------------------------------------------------

def test_schema_saves_and_loads_through_iap(window, tmp_path):
    _pose_class(window)
    window.all_images = []
    window.image_paths = {}
    window.current_project_file = str(tmp_path / "proj.iap")
    window.project_controller.save_project(show_message=False)

    data = json.loads((tmp_path / "proj.iap").read_text(encoding="utf-8"))
    person = next(c for c in data["classes"] if c["name"] == "person")
    assert person["keypoint_schema"]["names"] == _SCHEMA["names"]

    # Reload into a clean schema store.
    window.keypoint_schemas.clear()
    window.project_controller.load_project_data(data)
    assert window.keypoint_schemas["person"]["names"] == _SCHEMA["names"]


def test_malformed_schema_dropped_on_load(window):
    project_data = {
        "classes": [{"name": "bad", "color": "#1F77B4", "keypoint_schema": {"names": []}}],
        "images": [],
    }
    window.project_controller.load_project_data(project_data)
    assert "bad" not in window.keypoint_schemas  # malformed → dropped, no crash
    assert "bad" in window.class_mapping          # class itself still created


def test_rename_propagates_schema(window, monkeypatch):
    _pose_class(window, name="person")
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.class_controller.QInputDialog.getText",
        lambda *a, **k: ("human", True),
    )
    item = window.class_list.findItems("person", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)
    assert "person" not in window.keypoint_schemas
    assert window.keypoint_schemas["human"]["names"] == _SCHEMA["names"]


def test_delete_removes_schema(window, monkeypatch):
    _pose_class(window, name="person")
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.class_controller.QMessageBox.question",
        lambda *a, **k: __import__(
            "PyQt6.QtWidgets", fromlist=["QMessageBox"]
        ).QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.class_controller.QMessageBox.information",
        lambda *a, **k: None,
    )
    item = window.class_list.findItems("person", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.delete_class(item)
    assert "person" not in window.keypoint_schemas


def test_switching_to_non_pose_class_deactivates_keypoint_tool(window):
    # Activating Keypoint on "person" then selecting a schemaless class must
    # deactivate the tool — otherwise every click silently no-ops with no
    # feedback (the tool stays "active" but every press is rejected by the
    # missing-schema guard in KeypointTool.on_mouse_press).
    _pose_class(window, name="person")
    window.add_class("cell")  # normal (non-pose) class
    window.current_class = "person"
    window.activate_tool("keypoint")
    assert window.image_label.current_tool == "keypoint"

    cell_item = window.class_list.findItems("cell", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.on_class_selected(cell_item)

    assert window.image_label.current_tool is None
    assert not window.keypoint_button.isChecked()


# --- #44: a class is pose OR regular, not both (mixed-class guards) ---


def test_define_schema_blocked_on_class_with_plain_annotations(window, monkeypatch):
    # Turning a class that already holds polygons into a pose class is refused.
    window.add_class("cell")
    window.all_annotations["img.png"] = {
        "cell": [{"segmentation": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  "category_name": "cell"}]
    }
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.class_controller.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    item = window.class_list.findItems("cell", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.define_keypoint_schema(item)
    assert warned
    assert "cell" not in window.keypoint_schemas


def test_define_schema_allowed_editing_existing_pose_class(window, monkeypatch):
    # Editing the schema of an existing pose class stays allowed even if it is
    # a legacy-mixed class (has plain annotations) — only *new* conversions are
    # blocked, so names/skeleton can still be fixed.
    _pose_class(window, name="person")
    window.all_annotations["img.png"] = {
        "person": [{"segmentation": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    "category_name": "person"}]
    }
    from PyQt6.QtWidgets import QDialog

    constructed = {"yes": False}

    class _FakeDialog:
        DialogCode = QDialog.DialogCode

        def __init__(self, *a, **k):
            constructed["yes"] = True

        def exec(self):
            return QDialog.DialogCode.Rejected  # cancel — no schema change

    monkeypatch.setattr(
        "digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog.KeypointSchemaDialog",
        _FakeDialog,
    )
    item = window.class_list.findItems("person", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.define_keypoint_schema(item)
    assert constructed["yes"]  # the editor opened; the guard did not block


def test_shape_tool_blocked_on_pose_class(window, monkeypatch):
    _pose_class(window, name="person")
    window.enable_annotation_tools()  # tools are enabled once a class is selected
    monkeypatch.setattr(window.image_label, "check_unsaved_changes", lambda: True)
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.annotator_window.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    window.polygon_button.click()  # user clicks Polygon while a pose class is selected
    assert warned
    assert not window.polygon_button.isChecked()
    assert window.image_label.current_tool is None


def test_sam_box_blocked_on_pose_class(window, monkeypatch):
    _pose_class(window, name="person")
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.controllers.sam_controller.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    window.sam_box_button.setChecked(True)
    window.sam_controller.toggle_sam_box()
    assert warned
    assert not window.sam_box_button.isChecked()
    assert window.image_label.current_tool is None


def test_keypoint_tool_still_activates_on_pose_class(window, monkeypatch):
    _pose_class(window, name="person")
    window.enable_annotation_tools()
    monkeypatch.setattr(window.image_label, "check_unsaved_changes", lambda: True)
    window.keypoint_button.click()
    assert window.image_label.current_tool == "keypoint"
    assert window.keypoint_button.isChecked()


def test_selecting_pose_class_deactivates_active_shape_tool(window, monkeypatch):
    _pose_class(window, name="person")
    window.add_class("cell")  # normal class
    window.current_class = "cell"
    window.activate_tool("polygon")
    assert window.image_label.current_tool == "polygon"

    monkeypatch.setattr(window.image_label, "check_unsaved_changes", lambda: True)
    person_item = window.class_list.findItems("person", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.on_class_selected(person_item)
    assert window.image_label.current_tool is None


def test_dino_build_configs_skips_pose_class(window, monkeypatch):
    window.keypoint_schemas["person"] = copy.deepcopy(_SCHEMA)
    monkeypatch.setattr(
        window.dino_class_table, "get_class_configs",
        lambda: [
            {"name": "person", "box_thr": 0.3, "txt_thr": 0.25, "nms_thr": 0.5},
            {"name": "cell", "box_thr": 0.3, "txt_thr": 0.25, "nms_thr": 0.5},
        ],
    )
    monkeypatch.setattr(window.dino_phrase_panel, "get_phrases_for", lambda name: ["x"])
    configs = window.dino_controller._build_dino_class_configs()
    assert [c["name"] for c in configs] == ["cell"]
