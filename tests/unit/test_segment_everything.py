"""Segment Everything: proposal filtering, assignment and commit (issue #69).

The filters carry most of the tests, and deliberately so. An unprompted SAM
pass over a busy image can return several hundred masks, and without the area,
count and overlap limits the canvas is not merely cluttered, it is unusable.
Shipping the filters in the same change as the feature was a requirement, not
a nicety, and these tests are what holds that.

The second theme is that this must not become a *second* review mechanic
(ADR-015): the proposals travel through the same ``temp_annotations`` overlay
and the same Enter/Escape filter as DINO and SAM 3.
"""

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QKeyEvent

from src.digitalsreeni_image_annotator.core import mask_filters
from src.digitalsreeni_image_annotator.controllers.dino_controller import (
    REVIEW_SOURCES,
    DINOReviewEventFilter,
)
from src.digitalsreeni_image_annotator.controllers.segment_everything_controller import (
    SOURCE,
    TEMP_AUTO_CLASS,
)
from tests.canvas_fixtures import FakeCanvasContext, RecordingPainter, make_label


def _ring(x0, y0, side):
    return [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side]


def _proposal(x0, y0, side, score=0.9):
    return {"segmentation": _ring(x0, y0, side), "score": score}


# --- polygon IoU -----------------------------------------------------------


def test_identical_polygons_have_iou_one():
    ring = _ring(0, 0, 10)
    assert mask_filters.polygon_iou(ring, ring) == pytest.approx(1.0)


def test_disjoint_polygons_have_iou_zero():
    assert mask_filters.polygon_iou(_ring(0, 0, 10), _ring(100, 100, 10)) == 0.0


def test_half_overlap_is_one_third():
    """Two unit squares overlapping by half: intersection 50, union 150."""
    value = mask_filters.polygon_iou(_ring(0, 0, 10), _ring(5, 0, 10))
    assert value == pytest.approx(50 / 150)


def test_iou_of_degenerate_input_is_zero_not_an_error():
    assert mask_filters.polygon_iou([], _ring(0, 0, 10)) == 0.0
    assert mask_filters.polygon_iou([1, 2], _ring(0, 0, 10)) == 0.0


def test_self_intersecting_geometry_is_repaired_not_raised():
    bowtie = [0, 0, 10, 10, 10, 0, 0, 10]
    assert mask_filters.polygon_iou(bowtie, _ring(0, 0, 10)) >= 0.0


# --- filtering -------------------------------------------------------------


def test_speckle_below_the_minimum_area_is_dropped():
    kept, dropped = mask_filters.filter_mask_proposals(
        [_proposal(0, 0, 2)], 100, 100, min_area=100
    )
    assert kept == []
    assert dropped["too_small"] == 1


def test_a_background_sized_mask_is_dropped():
    """SAM reliably proposes the whole background; nobody wants it."""
    kept, dropped = mask_filters.filter_mask_proposals(
        [_proposal(0, 0, 100)], 100, 100, max_area_fraction=0.5
    )
    assert kept == []
    assert dropped["too_large"] == 1


def test_a_proposal_matching_an_existing_annotation_is_dropped():
    existing = [_ring(10, 10, 20)]
    kept, dropped = mask_filters.filter_mask_proposals(
        [_proposal(10, 10, 20)], 200, 200,
        existing_segmentations=existing, min_area=1,
    )
    assert kept == []
    assert dropped["overlapping"] == 1


def test_a_proposal_merely_near_an_existing_annotation_survives():
    existing = [_ring(10, 10, 20)]
    kept, _ = mask_filters.filter_mask_proposals(
        [_proposal(100, 100, 20)], 400, 400,
        existing_segmentations=existing, min_area=1,
    )
    assert len(kept) == 1


def test_the_count_cap_keeps_the_highest_scoring():
    """Capping before sorting would keep whichever masks the model happened to
    emit first, which is not the same thing as the best ones."""
    proposals = [_proposal(10 * i, 0, 8, score=i / 10) for i in range(10)]
    kept, dropped = mask_filters.filter_mask_proposals(
        proposals, 500, 500, min_area=1, max_candidates=3
    )
    assert [p["score"] for p in kept] == [0.9, 0.8, 0.7]
    assert dropped["over_limit"] == 7


def test_filters_are_off_the_hook_for_an_empty_input():
    kept, dropped = mask_filters.filter_mask_proposals([], 100, 100)
    assert kept == []
    assert all(count == 0 for count in dropped.values())
    assert mask_filters.filter_mask_proposals(None, 100, 100)[0] == []


def test_a_malformed_proposal_is_counted_not_crashed_on():
    kept, dropped = mask_filters.filter_mask_proposals(
        [{"segmentation": None}, {"segmentation": [1, 2]}], 100, 100
    )
    assert kept == []
    assert dropped["too_small"] == 2


def test_describe_dropped_names_the_reasons():
    text = mask_filters.describe_dropped(
        {"too_small": 3, "too_large": 0, "overlapping": 1, "over_limit": 0}
    )
    assert "3 below the minimum area" in text
    assert "1 overlapping existing annotations" in text
    assert "too_large" not in text


def test_describe_dropped_is_empty_when_nothing_was_dropped():
    assert mask_filters.describe_dropped(
        {"too_small": 0, "too_large": 0, "overlapping": 0, "over_limit": 0}
    ) == ""


# --- candidate assignment on the canvas -----------------------------------


@pytest.fixture
def label(qtbot):
    lbl = make_label(qtbot, ctx=FakeCanvasContext(current_class="cell"))
    lbl.class_colors = {"cell": QColor("#1F77B4"), TEMP_AUTO_CLASS: QColor("#888888")}
    lbl.temp_annotations = [
        {"segmentation": _ring(10, 10, 40), "category_name": TEMP_AUTO_CLASS,
         "score": 0.9, "source": SOURCE, "assigned_class": None, "temp": True},
        {"segmentation": _ring(100, 100, 40), "category_name": TEMP_AUTO_CLASS,
         "score": 0.8, "source": SOURCE, "assigned_class": None, "temp": True},
    ]
    return lbl


def test_only_unprompted_proposals_are_assignable(label):
    assert label.has_assignable_temp() is True
    label.temp_annotations = [
        {"segmentation": _ring(0, 0, 5), "category_name": "Temp-cell",
         "score": 0.5, "source": "dino"}
    ]
    assert label.has_assignable_temp() is False, (
        "a DINO proposal already knows its class"
    )


def test_click_assigns_the_active_class(label):
    assert label.assign_class_to_temp_at((20, 20)) is True
    assert label.temp_annotations[0]["assigned_class"] == "cell"
    assert label.temp_annotations[1]["assigned_class"] is None


def test_clicking_again_clears_the_assignment(label):
    """A mis-click is undone by repeating it, not by discarding the batch."""
    label.assign_class_to_temp_at((20, 20))
    label.assign_class_to_temp_at((20, 20))
    assert label.temp_annotations[0]["assigned_class"] is None


def test_shift_click_only_ever_assigns(label):
    label.assign_class_to_temp_at((20, 20))
    label.assign_class_to_temp_at((20, 20), additive=True)
    assert label.temp_annotations[0]["assigned_class"] == "cell"


def test_click_on_empty_space_assigns_nothing(label):
    assert label.assign_class_to_temp_at((180, 20)) is False
    assert all(a["assigned_class"] is None for a in label.temp_annotations)


def test_assignment_needs_an_active_class(qtbot):
    lbl = make_label(qtbot, ctx=FakeCanvasContext(current_class=None))
    lbl.temp_annotations = [
        {"segmentation": _ring(10, 10, 40), "category_name": TEMP_AUTO_CLASS,
         "score": 0.9, "source": SOURCE, "assigned_class": None}
    ]
    assert lbl.assign_class_to_temp_at((20, 20)) is False


def test_the_smallest_nested_proposal_wins(label):
    """An unprompted pass routinely nests a small mask inside a larger one; the
    small one would otherwise be unreachable."""
    small = {"segmentation": _ring(15, 15, 10), "category_name": TEMP_AUTO_CLASS,
             "score": 0.7, "source": SOURCE, "assigned_class": None}
    label.temp_annotations.append(small)
    label.assign_class_to_temp_at((20, 20))
    assert small["assigned_class"] == "cell"
    assert label.temp_annotations[0]["assigned_class"] is None


# --- rendering -------------------------------------------------------------


def test_assigned_and_unassigned_proposals_look_different(label):
    """Without this distinction a partially-reviewed batch is impossible to
    work through."""
    label.assign_class_to_temp_at((20, 20))

    painter = RecordingPainter()
    label.renderer.draw_temp_annotations(painter)

    styles = [args[0].style() for name, args in painter.calls if name == "setPen"]
    assert Qt.PenStyle.SolidLine in styles, "assigned proposal draws solid"
    assert Qt.PenStyle.DashLine in styles, "unassigned proposal stays dashed"


def test_an_assigned_proposal_is_labelled_with_its_class(label):
    label.assign_class_to_temp_at((20, 20))
    painter = RecordingPainter()
    label.renderer.draw_temp_annotations(painter)
    assert "cell 0.90" in painter.texts()
    assert f"{TEMP_AUTO_CLASS} 0.80" in painter.texts()


# --- the shared review gate (ADR-015) --------------------------------------


def test_the_source_is_registered_with_the_shared_review_filter():
    assert SOURCE in REVIEW_SOURCES
    assert "dino" in REVIEW_SOURCES and "sam3" in REVIEW_SOURCES


class _FakeLabelHolder:
    def __init__(self, temp):
        self.image_label = type("L", (), {"temp_annotations": temp})()
        self.accepted = 0
        self.rejected = 0

    def accept_dino_results(self):
        self.accepted += 1

    def reject_dino_results(self):
        self.rejected += 1


def test_enter_and_escape_reach_the_shared_accept_reject(qtbot):
    holder = _FakeLabelHolder(
        [{"segmentation": _ring(0, 0, 5), "source": SOURCE, "assigned_class": None}]
    )
    filt = DINOReviewEventFilter(None)
    filt.main_window = holder

    enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    escape = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )

    assert filt.eventFilter(None, enter) is True
    assert holder.accepted == 1
    assert filt.eventFilter(None, escape) is True
    assert holder.rejected == 1


def test_the_gate_stays_inert_without_reviewable_temp_annotations(qtbot):
    holder = _FakeLabelHolder([])
    filt = DINOReviewEventFilter(None)
    filt.main_window = holder
    enter = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    assert filt.eventFilter(None, enter) is False
    assert holder.accepted == 0
