"""Undo/redo of annotation edits (ADR-026).

Snapshot-based per-image history wired into AnnotationController. One real
offscreen ImageAnnotator (pattern from test_annotation_table.py); auto_save
is monkeypatched so nothing hits disk.
"""

import copy
import math

import pytest
from PyQt6.QtGui import QPixmap

from digitalsreeni_image_annotator.core.constants import ANNOT_COL_DETAIL


class _FakeEvent:
    from PyQt6.QtCore import Qt

    def modifiers(self):
        from PyQt6.QtCore import Qt

        return Qt.KeyboardModifier.NoModifier


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _square(x, y, s, number, cls="cell"):
    return {
        "segmentation": [x, y, x + s, y, x + s, y + s, x, y + s],
        "category_name": cls,
        "category_id": 1,
        "number": number,
    }


def _circle(cx, cy, r, n):
    seg = []
    for i in range(n):
        a = 2 * math.pi * i / n
        seg += [round(cx + r * math.cos(a)), round(cy + r * math.sin(a))]
    return seg


def _dense(number=1):
    return {"segmentation": _circle(50, 50, 30, 60), "category_name": "cell",
            "category_id": 1, "number": number}


def _seed(window, anns, monkeypatch):
    monkeypatch.setattr(window, "auto_save", lambda: None)
    window.image_file_name = "img.png"
    window.current_slice = None
    window.class_mapping = {"cell": 1}
    window.all_annotations = {"img.png": {"cell": copy.deepcopy(anns)}}
    window.image_label.annotations = copy.deepcopy(window.all_annotations["img.png"])
    window.update_annotation_list()
    return window.image_label.annotations["cell"]


def _cell(window):
    return window.all_annotations["img.png"].get("cell", [])


# --- delete / redo ---------------------------------------------------------


def test_delete_then_undo_then_redo(window, monkeypatch):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    assert len(_cell(window)) == 1

    ac.undo()
    assert len(_cell(window)) == 2

    ac.redo()
    assert len(_cell(window)) == 1


def test_new_edit_truncates_redo(window, monkeypatch):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    ac.undo()                                  # redo now available
    assert ac.history.can_redo("img.png")

    ac.apply_canvas_selection([a2], "replace")
    ac.delete_selected_annotations()           # a fresh edit
    assert not ac.history.can_redo("img.png")


# --- merge -----------------------------------------------------------------


def test_merge_then_undo_restores_originals(window, monkeypatch):
    a1, a2 = _square(0, 0, 20, 1), _square(10, 0, 20, 2)  # overlapping → connected
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1, a2], "replace")
    ac.merge_annotations()
    assert len(_cell(window)) == 1            # union replaced both

    ac.undo()
    assert len(_cell(window)) == 2            # both originals back


# --- create ----------------------------------------------------------------


def test_create_via_sam_accept_then_undo(window, monkeypatch):
    _seed(window, [], monkeypatch)
    il = window.image_label
    il.temp_sam_prediction = {
        "segmentation": [0, 0, 10, 0, 10, 10, 0, 10],
        "category_id": 1,
        "category_name": "cell",
        "score": 0.9,
    }
    window.sam_controller.accept_sam_prediction()
    assert len(_cell(window)) == 1

    window.annotation_controller.undo()
    assert _cell(window) == []


# --- detail-% coalescing ---------------------------------------------------


def test_detail_drag_coalesces_to_one_undo(window, monkeypatch):
    _seed(window, [_dense(1)], monkeypatch)
    ac = window.annotation_controller
    spin = window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL)

    for v in (90, 70, 50, 30):
        spin.setValue(v)
    assert _cell(window)[0]["detail_pct"] == 30

    ac.undo()                                  # one entry restores pre-drag state
    assert _cell(window)[0].get("detail_pct", 100) == 100
    assert not ac.history.can_undo("img.png")  # coalesced into a single entry


# --- bbox move / scale (deferred-gesture baseline) -------------------------


def test_bbox_resize_then_undo_restores_geometry(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    il = window.image_label
    il.original_pixmap = QPixmap(200, 200)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    ac = window.annotation_controller
    ac.apply_canvas_selection([live], "replace")
    before = list(_cell(window)[0]["segmentation"])

    bb = il._annotation_bbox(live)
    il._begin_shape_edit(live, "resize", "br", (bb[2], bb[3]))
    il._update_bbox_drag((bb[2] + 10, bb[3] + 10))
    il._commit_bbox_drag((bb[2] + 10, bb[3] + 10), _FakeEvent())
    assert _cell(window)[0]["segmentation"] != before

    ac.undo()
    assert _cell(window)[0]["segmentation"] == before


def test_aborted_bbox_leaves_no_history(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    il = window.image_label
    il.original_pixmap = QPixmap(200, 200)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    ac = window.annotation_controller
    ac.apply_canvas_selection([live], "replace")

    bb = il._annotation_bbox(live)
    il._begin_shape_edit(live, "resize", "br", (bb[2], bb[3]))  # captures baseline
    il._update_bbox_drag((bb[2] + 10, bb[3] + 10))
    il._cancel_bbox_drag()                                       # Esc: no commit
    assert not ac.history.can_undo("img.png")


# --- isolation / guards / persistence --------------------------------------


def test_per_image_stacks_isolated(window, monkeypatch):
    a1 = _square(0, 0, 10, 1)
    _seed(window, [a1], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    assert ac.history.can_undo("img.png")

    window.image_file_name = "other.png"
    window.all_annotations["other.png"] = {"cell": []}
    assert not ac.history.can_undo("other.png")
    ac.undo()                                  # no-op on the other image
    assert window.all_annotations["other.png"] == {"cell": []}


def test_undo_noop_during_draw(window, monkeypatch):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    assert len(_cell(window)) == 1

    window.image_label.drawing_polygon = True   # a gesture is in flight
    ac.undo()
    assert len(_cell(window)) == 1              # blocked

    window.image_label.drawing_polygon = False
    ac.undo()
    assert len(_cell(window)) == 2


def test_undo_persists(window, monkeypatch):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()

    calls = []
    monkeypatch.setattr(window, "auto_save", lambda: calls.append(1))
    ac.undo()
    assert calls                                # undo wrote through auto_save


def test_undo_blocked_mid_paint_stroke(window, monkeypatch):
    import numpy as np

    a1 = _square(0, 0, 10, 1)
    _seed(window, [a1], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    assert len(_cell(window)) == 0

    # An uncommitted paint stroke is in flight → undo must be a no-op, or it
    # would restore a snapshot while a deferred baseline is still pending.
    window.image_label.temp_paint_mask = np.zeros((10, 10), dtype=np.uint8)
    ac.undo()
    assert len(_cell(window)) == 0          # blocked

    window.image_label.temp_paint_mask = None
    ac.undo()
    assert len(_cell(window)) == 1          # now restored


def test_undo_restores_snapshot_numbers_verbatim(window, monkeypatch):
    # Delete doesn't renumber, so deleting the middle leaves a gap (1, 3).
    # Undo must restore that exact numbering on BOTH the table UserRole and the
    # model — _restore_snapshot must not renumber only one copy.
    from PyQt6.QtCore import Qt

    from digitalsreeni_image_annotator.core.constants import ANNOT_COL_ID

    a1, a2, a3 = _square(0, 0, 10, 1), _square(50, 0, 10, 2), _square(100, 0, 10, 3)
    _seed(window, [a1, a2, a3], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a2], "replace")
    ac.delete_selected_annotations()        # -> [1, 3], a gap
    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()        # -> [3]
    ac.undo()                               # restore [1, 3]

    tbl = window.annotation_list
    table_nums = sorted(
        tbl.item(r, ANNOT_COL_ID).data(Qt.ItemDataRole.UserRole)["number"]
        for r in range(tbl.rowCount())
    )
    model_nums = sorted(a["number"] for a in _cell(window))
    assert table_nums == model_nums == [1, 3]   # parity + verbatim restore


def _poly():
    return {"segmentation": [10, 10, 90, 10, 90, 90, 10, 90],
            "category_name": "cell", "category_id": 1, "number": 1}


def _arm_edit_canvas(window):
    il = window.image_label
    il.original_pixmap = QPixmap(100, 100)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    return il


def test_vertex_edit_commit_then_undo(window, monkeypatch):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    _seed(window, [_poly()], monkeypatch)
    il = _arm_edit_canvas(window)
    ac = window.annotation_controller
    before = list(_cell(window)[0]["segmentation"])

    assert il.start_polygon_edit((50, 50)) is not None   # double-click enters edit
    il.editing_point_index = 0
    il.handle_editing_move((20, 20))                      # drag first vertex
    assert il.editing_polygon["segmentation"][0:2] == [20, 20]

    il.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    )
    assert il.editing_polygon is None
    assert _cell(window)[0]["segmentation"][0:2] == [20, 20]   # committed + saved
    assert ac.history.can_undo("img.png")

    ac.undo()
    assert _cell(window)[0]["segmentation"] == before          # reverted


def test_vertex_edit_escape_reverts_without_history(window, monkeypatch):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    _seed(window, [_poly()], monkeypatch)
    il = _arm_edit_canvas(window)
    ac = window.annotation_controller
    before = list(il.annotations["cell"][0]["segmentation"])

    il.start_polygon_edit((50, 50))
    il.editing_point_index = 0
    il.handle_editing_move((25, 25))
    assert il.editing_polygon["segmentation"][0:2] == [25, 25]

    il.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )
    assert il.editing_polygon is None
    assert list(il.annotations["cell"][0]["segmentation"]) == before  # Esc reverts
    assert not ac.history.can_undo("img.png")                         # no entry


def test_clear_history_drops_stacks(window, monkeypatch):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2], monkeypatch)
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.delete_selected_annotations()
    assert ac.history.can_undo("img.png")

    ac.clear_history()
    assert not ac.history.can_undo("img.png")
