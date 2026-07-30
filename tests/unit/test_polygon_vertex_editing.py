"""Vertex insertion and removal during polygon edit (issue #68, ADR-023).

Before this, a polygon's vertex count was fixed at creation: vertices could be
dragged, the whole shape scaled and moved, but a mask that was *almost* right
had to be deleted and redrawn. On a SAM mask with a few hundred vertices, not
being able to delete one stray point is a real limitation.

The test that matters most here is
``test_detail_pct_does_not_revert_an_inserted_vertex``: the Detail-%
simplification baseline (``segmentation_raw``, ADR-025) is captured lazily and
is what the spinbox re-simplifies *from*. Leave it stale after a vertex-count
change and the next Detail-% drag silently reverts the user's edit — a bug that
would be nearly impossible to diagnose from a report.
"""

import pytest
from PyQt6.QtCore import QPoint, Qt

from src.digitalsreeni_image_annotator.utils import calculate_area, simplify_polygon
from src.digitalsreeni_image_annotator.widgets import edit_gestures
from tests.canvas_fixtures import bbox, make_label, pose, square


@pytest.fixture
def label(qtbot):
    lbl = make_label(qtbot, width=200, height=200)
    lbl.show()
    qtbot.waitExposed(lbl)
    lbl.offset_x = lbl.offset_y = 0
    return lbl


def _catch(signal):
    seen = []
    signal.connect(lambda *args: seen.append(args if args else True))
    return seen


def _enter_edit(qtbot, label, annotation, at):
    label.annotations = {annotation["category_name"]: [annotation]}
    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(*at))
    assert label.editing_polygon is annotation


# --- pure geometry ---------------------------------------------------------


def test_project_onto_segment_finds_the_perpendicular_foot():
    point, distance = edit_gestures.project_onto_segment((5, 10), (0, 0), (10, 0))
    assert point == pytest.approx((5, 0))
    assert distance == pytest.approx(10)


def test_projection_clamps_past_the_segment_ends():
    point, _ = edit_gestures.project_onto_segment((-50, 0), (0, 0), (10, 0))
    assert point == pytest.approx((0, 0))
    point, _ = edit_gestures.project_onto_segment((99, 0), (0, 0), (10, 0))
    assert point == pytest.approx((10, 0))


def test_projection_survives_a_degenerate_edge():
    """Duplicate consecutive vertices exist in imported data; a zero-length
    segment must not divide by zero."""
    point, distance = edit_gestures.project_onto_segment((3, 4), (0, 0), (0, 0))
    assert point == (0, 0)
    assert distance == pytest.approx(5)


def test_closest_edge_returns_the_insertion_index():
    ring = [0, 0, 100, 0, 100, 100, 0, 100]
    index, point = edit_gestures.closest_edge(ring, (50, 2), tolerance=10)
    assert index == 1, "edge 0->1 inserts at position 1"
    assert point == pytest.approx((50, 0))


def test_closest_edge_covers_the_closing_edge():
    ring = [0, 0, 100, 0, 100, 100, 0, 100]
    index, point = edit_gestures.closest_edge(ring, (2, 50), tolerance=10)
    assert index == 4, "the last->first edge appends"
    assert point == pytest.approx((0, 50))


def test_closest_edge_respects_the_tolerance():
    ring = [0, 0, 100, 0, 100, 100, 0, 100]
    assert edit_gestures.closest_edge(ring, (50, 40), tolerance=10) is None


def test_closest_edge_picks_the_nearest_of_several():
    ring = [0, 0, 100, 0, 100, 100, 0, 100]
    index, _ = edit_gestures.closest_edge(ring, (98, 50), tolerance=10)
    assert index == 2, "the right-hand edge, not the far left one"


def test_insert_and_remove_are_inverses():
    ring = [0, 0, 100, 0, 100, 100]
    grown = edit_gestures.insert_vertex(ring, 1, (50, 0))
    assert grown == [0, 0, 50, 0, 100, 0, 100, 100]
    assert edit_gestures.remove_vertex(grown, 1) == ring


def test_can_remove_vertex_stops_at_the_minimum():
    assert edit_gestures.can_remove_vertex([0, 0, 1, 0, 1, 1, 0, 1]) is True
    assert edit_gestures.can_remove_vertex([0, 0, 1, 0, 1, 1]) is False


def test_invalidate_raw_polygon_resets_the_detail_baseline():
    annotation = {"segmentation": [0, 0, 1, 0, 1, 1],
                  "segmentation_raw": [0, 0, 2, 0, 2, 2], "detail_pct": 40}
    edit_gestures.invalidate_raw_polygon(annotation)
    assert "segmentation_raw" not in annotation
    assert annotation["detail_pct"] == 100


# --- insertion through the gesture -----------------------------------------


def test_double_click_on_an_edge_inserts_a_vertex(qtbot, label):
    annotation = square(40, 40, 80)          # 4 vertices, edges on the axes
    _enter_edit(qtbot, label, annotation, (80, 80))

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(80, 40))

    assert len(annotation["segmentation"]) // 2 == 5


def test_the_inserted_vertex_lies_on_the_original_edge(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))
    before = calculate_area(annotation)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(80, 43))

    # A point *on* the outline adds no area; the outline must not visibly jump.
    assert calculate_area(annotation) == pytest.approx(before, abs=1.0)


def test_double_click_away_from_any_edge_does_not_insert(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(80, 80))

    assert len(annotation["segmentation"]) // 2 == 4


def test_insert_syncs_the_derived_bbox(qtbot, label):
    annotation = square(40, 40, 80)
    annotation["bbox"] = [0, 0, 1, 1]        # deliberately stale
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))

    assert annotation["bbox"] == pytest.approx([40, 40, 80, 80])


# --- removal ---------------------------------------------------------------


@pytest.mark.parametrize(
    "modifier",
    [Qt.KeyboardModifier.AltModifier, Qt.KeyboardModifier.ShiftModifier],
    ids=["alt", "shift-legacy"],
)
def test_modified_click_on_a_vertex_removes_it(qtbot, label, modifier):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    qtbot.mouseClick(label, Qt.MouseButton.LeftButton, modifier, QPoint(40, 40))

    assert len(annotation["segmentation"]) // 2 == 3
    assert annotation["segmentation"][:2] != [40, 40]


def test_removal_is_refused_at_three_vertices(qtbot, label):
    annotation = {
        "segmentation": [40, 40, 120, 40, 120, 120],
        "category_name": "cell",
        "number": 1,
    }
    _enter_edit(qtbot, label, annotation, (100, 60))

    assert label.remove_editing_vertex(0) is False
    assert len(annotation["segmentation"]) // 2 == 3


def test_plain_click_on_a_vertex_starts_a_drag_instead(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    qtbot.mousePress(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))

    assert label.editing_point_index == 0
    assert len(annotation["segmentation"]) // 2 == 4, "a plain click must not remove"


def test_plain_click_on_an_edge_does_nothing(qtbot, label):
    """Edge insertion moved to double-click; if the *first* click of that
    double-click also inserted, every insertion would add two vertices."""
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.handle_editing_click((80, 40), _AltlessEvent())

    assert len(annotation["segmentation"]) // 2 == 4
    assert label.editing_point_index is None


class _AltlessEvent:
    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


# --- the Detail-% interaction (the one that would bite) --------------------


def test_insert_invalidates_the_detail_baseline(qtbot, label):
    annotation = square(40, 40, 80)
    annotation["segmentation_raw"] = list(annotation["segmentation"])
    annotation["detail_pct"] = 50
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))

    assert "segmentation_raw" not in annotation
    assert annotation["detail_pct"] == 100


def test_detail_pct_does_not_revert_an_inserted_vertex(qtbot, label):
    """End-to-end version of the trap: insert a vertex, then simplify the way
    the Detail-% spinbox does, and assert the edit survives.

    Simulated rather than driven through the spinbox because the spinbox lives
    on the main-window table; the invariant under test belongs to the data.
    """
    annotation = square(40, 40, 80)
    annotation["segmentation_raw"] = list(annotation["segmentation"])
    annotation["detail_pct"] = 60
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))
    edited = list(annotation["segmentation"])

    # What on_detail_pct_changed does at 100 %: restore from the raw copy.
    raw = annotation.get("segmentation_raw")
    restored = list(raw) if raw else list(annotation["segmentation"])

    assert restored == edited, "Detail-% reverted the inserted vertex"


def test_removal_invalidates_the_detail_baseline_too(qtbot, label):
    annotation = square(40, 40, 80)
    annotation["segmentation_raw"] = list(annotation["segmentation"])
    _enter_edit(qtbot, label, annotation, (80, 80))

    assert label.remove_editing_vertex(0) is True
    assert "segmentation_raw" not in annotation


def test_simplify_from_the_new_baseline_keeps_the_edited_vertex_count(qtbot, label):
    """After invalidation the *edited* polygon becomes the baseline, so a later
    simplification is derived from what the user now sees."""
    annotation = square(20, 20, 160)
    _enter_edit(qtbot, label, annotation, (100, 100))
    for x in (60, 100, 140):
        label.insert_editing_vertex((x, 20))

    edited = list(annotation["segmentation"])
    assert len(edited) // 2 == 7
    # simplify_polygon(raw, 100) is the identity, and the raw is now the edit.
    assert simplify_polygon(edited, 100) == edited


# --- Esc / undo semantics --------------------------------------------------


def test_escape_reverts_an_insertion(qtbot, label):
    annotation = square(40, 40, 80)
    original = list(annotation["segmentation"])
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))
    qtbot.keyClick(label, Qt.Key.Key_Escape)

    assert annotation["segmentation"] == original


def test_escape_restores_the_detail_baseline(qtbot, label):
    """Cancelling must not cost the user their simplification state either."""
    annotation = square(40, 40, 80)
    annotation["segmentation_raw"] = [1, 1, 2, 2, 3, 3]
    annotation["detail_pct"] = 35
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))
    qtbot.keyClick(label, Qt.Key.Key_Escape)

    assert annotation["segmentation_raw"] == [1, 1, 2, 2, 3, 3]
    assert annotation["detail_pct"] == 35


def test_escape_after_an_insert_commits_nothing(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))
    committed = _catch(label.polygonEditCommitted)

    label.insert_editing_vertex((80, 40))
    qtbot.keyClick(label, Qt.Key.Key_Escape)

    assert committed == [], "an aborted session must leave no history entry"


def test_enter_commits_the_session_once(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))
    committed = _catch(label.polygonEditCommitted)

    label.insert_editing_vertex((80, 40))
    label.insert_editing_vertex((40, 80))
    qtbot.keyClick(label, Qt.Key.Key_Return)

    assert committed == [True], "one undo step for the whole edit session"
    assert label.editing_polygon is None


def test_a_mid_edit_change_refreshes_without_committing(qtbot, label):
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))
    refreshed = _catch(label.polygonGeometryChanged)
    committed = _catch(label.polygonEditCommitted)

    label.insert_editing_vertex((80, 40))

    assert refreshed == [True], "the table must refresh immediately"
    assert committed == [], "but the gesture is not over yet"


def test_enter_syncs_the_bbox_after_a_plain_vertex_drag(qtbot, label):
    annotation = square(40, 40, 80)
    annotation["bbox"] = [40, 40, 80, 80]
    _enter_edit(qtbot, label, annotation, (80, 80))

    annotation["segmentation"][0] = 10      # drag the first vertex left
    annotation["segmentation"][1] = 10
    qtbot.keyClick(label, Qt.Key.Key_Return)

    assert annotation["bbox"][0] == pytest.approx(10)


# --- the undo baseline across a session switch (senior-review finding) ------


class _RecordingAnnotationController:
    """The two ADR-026 halves that matter here, with no main window.

    ``capture_edit_baseline`` overwrites unconditionally in the real
    controller, which is exactly the behaviour under test.
    """

    def __init__(self):
        self.pending = None
        self.pushed = []
        self.state = "initial"

    def capture_edit_baseline(self):
        self.pending = self.state

    def commit_edit_baseline(self):
        if self.pending is None:
            return
        self.pushed.append(self.pending)
        self.pending = None

    def sync_polygon_geometry(self):
        # Mirrors the real slot: persists, does NOT push history.
        self.state = "persisted"


def _wire(label):
    controller = _RecordingAnnotationController()
    label.editBaselineRequested.connect(controller.capture_edit_baseline)
    label.polygonEditCommitted.connect(controller.commit_edit_baseline)
    label.polygonGeometryChanged.connect(controller.sync_polygon_geometry)
    return controller


def test_switching_polygon_mid_session_does_not_lose_the_undo_entry(qtbot, label):
    """The hole the senior review found.

    An insert is persisted immediately by ``sync_polygon_geometry`` but leaves
    the undo baseline pending until the session ends. Double-clicking a
    *different* polygon used to call ``start_polygon_edit`` straight away,
    which re-emitted ``editBaselineRequested`` and overwrote the pending
    baseline with the **post-insert** state — making the first polygon's
    already-saved edit permanently un-undoable.
    """
    first = square(20, 20, 40)
    second = square(120, 120, 40, number=2)
    label.annotations = {"cell": [first, second]}
    controller = _wire(label)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    assert label.editing_polygon is first
    # Mid-edge, not on a corner: an insert exactly on a vertex is refused.
    assert label.insert_editing_vertex((40, 20)) is True
    assert controller.state == "persisted", "the insert should be saved"

    # Now start a session on the other polygon.
    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(140, 140))

    assert controller.pushed == ["initial"], (
        "the first polygon's edit was persisted with no history entry"
    )
    assert label.editing_polygon is second


def test_enter_and_the_session_switch_push_the_same_baseline(qtbot, label):
    """Both routes go through finish_polygon_edit, so they cannot drift on
    which of them records history."""
    annotation = square(20, 20, 40)
    label.annotations = {"cell": [annotation]}
    controller = _wire(label)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    assert label.insert_editing_vertex((40, 20)) is True
    qtbot.keyClick(label, Qt.Key.Key_Return)

    assert controller.pushed == ["initial"]
    assert controller.pending is None


def test_leaving_edit_mode_on_a_switch_still_pushes_the_baseline(qtbot, label):
    """``exit_editing_mode`` is called on every image and slice switch.

    It used to just clear the state, which reopened the lost-baseline hole from
    the other side: the insert is already persisted by
    ``sync_polygon_geometry``, so exiting without emitting
    ``polygonEditCommitted`` left a saved edit with no history entry.
    """
    annotation = square(20, 20, 40)
    label.annotations = {"cell": [annotation]}
    controller = _wire(label)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    assert label.insert_editing_vertex((40, 20)) is True

    label.exit_editing_mode()

    assert controller.pushed == ["initial"]
    assert label.editing_polygon is None


def test_leaving_edit_mode_clamps_the_polygon(qtbot, label):
    """The switch path used to skip clamp_segmentation, so a vertex dragged out
    of bounds at the moment of a switch was persisted unclamped (ADR-024)."""
    annotation = square(20, 20, 40)
    label.annotations = {"cell": [annotation]}
    _wire(label)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    label.editing_polygon["segmentation"][0] = 5000  # drag a vertex far right
    label.exit_editing_mode()

    assert max(annotation["segmentation"][0::2]) <= 200


def test_leaving_edit_mode_without_a_session_is_a_no_op(label):
    controller = _wire(label)
    label.exit_editing_mode()
    assert controller.pushed == []


# --- P2 fixes from the senior review ----------------------------------------


def test_double_clicking_a_vertex_does_not_plant_a_duplicate(qtbot, label):
    """A vertex lies on both adjacent edges at perpendicular distance zero, so
    an unguarded insert always "succeeds" there."""
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    assert label.insert_editing_vertex((40, 40)) is False
    assert len(annotation["segmentation"]) // 2 == 4


def test_double_clicking_a_vertex_does_not_end_the_session(qtbot, label):
    """Through the real event, not the helper: the refused insert must return
    early rather than fall through to finish-and-restart. A vertex sits on the
    polygon boundary where point_in_polygon is unreliable, so falling through
    could end the session outright on what the user meant as a no-op."""
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))

    assert label.editing_polygon is annotation
    assert len(annotation["segmentation"]) // 2 == 4


def test_the_inserted_vertex_is_immediately_draggable(qtbot, label):
    """Matches what the pre-#68 edge-click path did; inserting a point you then
    have to go and grab is a step backwards."""
    annotation = square(40, 40, 80)
    _enter_edit(qtbot, label, annotation, (80, 80))

    label.insert_editing_vertex((80, 40))

    assert label.editing_point_index is not None
    index = label.editing_point_index
    segmentation = annotation["segmentation"]
    assert (segmentation[index * 2], segmentation[index * 2 + 1]) == pytest.approx(
        (80, 40)
    )


# --- shapes the gesture must never reach -----------------------------------


def test_vertex_edit_is_unreachable_on_a_bbox_only_annotation(qtbot, label):
    """An imported bbox carries ``"segmentation": None``; slicing that raises,
    so the guard is a crash fix as much as a scoping one."""
    annotation = bbox(40, 40, 80, 80)
    annotation["segmentation"] = None
    label.annotations = {"cell": [annotation]}

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(80, 80))

    assert label.editing_polygon is None


def test_vertex_edit_is_unreachable_on_a_pose_instance(qtbot, label):
    """K is locked by the class schema (ADR-029) — inserting a point into a
    pose is a schema change, not an edit."""
    instance = pose([(50, 50, 2), (70, 70, 2), (90, 50, 2)])
    label.annotations = {"person": [instance]}

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(70, 60))

    assert label.editing_polygon is None
