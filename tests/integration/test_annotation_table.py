"""Annotations table + reversible polygon simplification (issue #24).

The bottom-left Annotations panel is a QTableWidget (ID | Class | Area |
Detail %). Each row's Detail % spinbox re-simplifies that annotation's polygon
from a lazily-captured raw copy; 100 % restores raw exactly. The simplified
polygon persists (.iap) and is what exports emit; the raw is kept for reverting.

One real offscreen ImageAnnotator; no model weights.
"""

import copy
import json
import math

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QSpinBox

from digitalsreeni_image_annotator.core import image_utils
from digitalsreeni_image_annotator.core.constants import (
    ANNOT_COL_AREA,
    ANNOT_COL_CLASS,
    ANNOT_COL_DETAIL,
    ANNOT_COL_ID,
)
from digitalsreeni_image_annotator.utils import calculate_area


class _FakeEvent:
    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _arm_canvas(window, live):
    """Set up the image label for #40-style handle edits + select the mask."""
    il = window.image_label
    il.original_pixmap = QPixmap(100, 100)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    window.annotation_controller.apply_canvas_selection([live], "replace")
    return il


def _circle(cx, cy, r, n):
    seg = []
    for i in range(n):
        a = 2 * math.pi * i / n
        seg += [round(cx + r * math.cos(a)), round(cy + r * math.sin(a))]
    return seg


def _seed(window, anns, monkeypatch):
    monkeypatch.setattr(window, "auto_save", lambda: None)
    window.image_file_name = "img.png"
    window.current_slice = None
    window.class_mapping = {"cell": 1}
    window.all_annotations = {"img.png": {"cell": copy.deepcopy(anns)}}
    window.image_label.annotations = copy.deepcopy(window.all_annotations["img.png"])
    window.update_annotation_list()
    return window.image_label.annotations["cell"]


def _dense(number=1):
    return {"segmentation": _circle(50, 50, 30, 60), "category_name": "cell",
            "category_id": 1, "number": number}


def test_table_populates_columns(window, monkeypatch):
    _seed(window, [_dense(1)], monkeypatch)
    tbl = window.annotation_list
    assert tbl.rowCount() == 1
    assert tbl.item(0, ANNOT_COL_ID).text() == "1"
    assert tbl.item(0, ANNOT_COL_CLASS).text() == "cell"
    float(tbl.item(0, ANNOT_COL_AREA).text())            # area is numeric
    spin = tbl.cellWidget(0, ANNOT_COL_DETAIL)
    assert isinstance(spin, QSpinBox) and spin.value() == 100


def test_detail_change_simplifies_live_and_refreshes_area(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    tbl = window.annotation_list
    raw_len = len(live["segmentation"])

    tbl.cellWidget(0, ANNOT_COL_DETAIL).setValue(30)      # fires the handler

    assert len(live["segmentation"]) < raw_len            # thinned
    assert live["segmentation_raw"] and len(live["segmentation_raw"]) == raw_len
    assert live["detail_pct"] == 30
    # Area cell tracks the (now simplified) polygon.
    assert tbl.item(0, ANNOT_COL_AREA).text() == f"{calculate_area(live):.2f}"


def test_back_to_100_restores_raw_exactly(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    spin = window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL)
    spin.setValue(25)
    assert len(live["segmentation"]) < len(live["segmentation_raw"])

    spin.setValue(100)
    assert live["segmentation"] == live["segmentation_raw"]  # raw restored exactly
    assert live["detail_pct"] == 100


def test_simplified_persists_and_exports_simplified(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL).setValue(40)

    # Persistence: project save does ann.copy() -> convert_to_serializable ->
    # json. Both new keys must survive the round-trip.
    roundtripped = json.loads(json.dumps(image_utils.convert_to_serializable(live)))
    assert roundtripped["detail_pct"] == 40
    assert roundtripped["segmentation_raw"] == live["segmentation_raw"]

    # Export emits the *effective* (simplified) polygon, not the raw.
    coco = window.annotation_controller.create_coco_annotation(live, 1, 1)
    assert coco["segmentation"] == [live["segmentation"]]
    assert coco["segmentation"][0] != live["segmentation_raw"]


def test_create_coco_annotation_delegates_and_stays_keypoint_aware(window):
    """AnnotationController.create_coco_annotation now delegates to the live
    io.export_formats implementation (issue #35 PR-2 senior-review fix) —
    regression guard against it drifting back into a keypoint-blind copy."""
    ann = {
        "keypoints": [10, 10, 2, 20, 20, 1],
        "num_keypoints": 2,
        "bbox": [5, 5, 20, 20],
        "category_id": 1,
        "category_name": "person",
    }

    coco = window.annotation_controller.create_coco_annotation(ann, 1, 1)

    assert coco["keypoints"] == [10, 10, 2, 20, 20, 1]
    assert "segmentation" not in coco


def test_reshape_invalidates_simplification_baseline(window, monkeypatch):
    # #24 × #40 seam: thinning then reshaping a polygon must reset the raw
    # baseline, so a later Detail %=100 can't silently revert the reshape.
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    il = _arm_canvas(window, live)
    window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL).setValue(40)
    assert live.get("segmentation_raw")               # raw captured by the thin

    bb = il._annotation_bbox(live)                     # (x0, y0, x1, y1)
    il._begin_shape_edit(live, "resize", "br", (bb[2], bb[3]))
    il._update_bbox_drag((bb[2] + 8, bb[3] + 8))
    il._commit_bbox_drag((bb[2] + 8, bb[3] + 8), _FakeEvent())

    assert not live.get("segmentation_raw")           # baseline invalidated
    assert live.get("detail_pct") == 100              # edited geometry is the new raw


def test_detail_change_keeps_selection_resolvable_for_handle_drag(window, monkeypatch):
    # #24 × #40 seam: after a detail change, the canvas selection must point at
    # the mutated live object so the overlay + a later handle drag resolve it.
    # The bug only manifests for a LIST-driven selection, where
    # update_highlighted_annotations stores a UserRole *copy* (not the live
    # object) — so select through the table, not via apply_canvas_selection.
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    il = window.image_label
    il.original_pixmap = QPixmap(100, 100)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    window.annotation_list.selectRow(0)                # → update_highlighted_annotations
    window.annotation_controller.update_highlighted_annotations()
    assert il.highlighted_annotations[0] is not live   # a UserRole copy, by value

    window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL).setValue(30)

    # The re-point loop must now make the selection the live, mutated object.
    assert il.highlighted_annotations[0] is live
    shape = il._single_selected_shape()
    assert il._live_annotation(shape) is live          # not a stale pre-thin copy


def test_bbox_only_row_has_disabled_spinbox(window, monkeypatch):
    _seed(window, [{"bbox": [10, 10, 40, 40], "segmentation": None,
                    "category_name": "cell", "category_id": 1, "number": 1}],
          monkeypatch)
    spin = window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL)
    assert isinstance(spin, QSpinBox) and not spin.isEnabled()
