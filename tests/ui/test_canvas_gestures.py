"""Real-Qt-event gesture tests for the canvas (issue #77, ADR-023).

These drive ``ImageLabel`` through ``qtbot.mousePress`` / ``mouseMove`` /
``mouseRelease`` rather than calling the handlers directly, so the dispatch
chain in ``mousePressEvent`` -- which branch wins, and in what priority order
-- is covered as well as the gesture logic itself. That dispatch is exactly
what the #40/#35 invariants live in and where a regression is least visible by
inspection.

Assertions are synchronous (no ``qtbot.wait``): every gesture here mutates
state or emits during the event call, so waiting would only add flake.
"""

import pytest
from PyQt6.QtCore import QPoint, Qt

from tests.canvas_fixtures import FakeCanvasContext, make_label, square


@pytest.fixture
def label(qtbot):
    lbl = make_label(qtbot, width=200, height=200)
    lbl.show()
    qtbot.waitExposed(lbl)
    # Showing the widget lays it out and re-centres the image; pin the offset
    # back to zero so a widget position is an image position.
    lbl.offset_x = 0
    lbl.offset_y = 0
    return lbl


def _catch(signal):
    seen = []
    signal.connect(lambda *args: seen.append(args if args else True))
    return seen


def _press_drag_release(qtbot, label, start, end, modifier=Qt.KeyboardModifier.NoModifier):
    qtbot.mousePress(label, Qt.MouseButton.LeftButton, modifier, QPoint(*start))
    qtbot.mouseMove(label, QPoint(*end))
    qtbot.mouseRelease(label, Qt.MouseButton.LeftButton, modifier, QPoint(*end))


# --- rubber-band selection -------------------------------------------------


def test_rubber_band_selects_the_enclosed_masks(qtbot, label):
    inside = square(20, 20, 30)
    outside = square(150, 150, 30, number=2)
    label.annotations = {"cell": [inside, outside]}
    selections = _catch(label.canvasSelectionChanged)

    _press_drag_release(qtbot, label, (10, 10), (100, 100))

    assert len(selections) == 1
    annotations, mode = selections[0]
    assert mode == "replace"
    assert annotations == [inside]


def test_shift_drag_adds_to_the_selection(qtbot, label):
    label.annotations = {"cell": [square(20, 20, 30)]}
    selections = _catch(label.canvasSelectionChanged)

    _press_drag_release(qtbot, label, (10, 10), (100, 100),
                        Qt.KeyboardModifier.ShiftModifier)

    assert selections[0][1] == "add"


def test_plain_click_replaces_and_shift_click_toggles(qtbot, label):
    annotation = square(20, 20, 30)
    label.annotations = {"cell": [annotation]}
    selections = _catch(label.canvasSelectionChanged)

    qtbot.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(30, 30))
    assert selections[-1] == ([annotation], "replace")

    qtbot.mouseClick(label, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ShiftModifier, QPoint(30, 30))
    assert selections[-1] == ([annotation], "toggle")


def test_click_on_empty_space_clears_the_selection(qtbot, label):
    label.annotations = {"cell": [square(20, 20, 30)]}
    selections = _catch(label.canvasSelectionChanged)
    qtbot.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(180, 180))
    assert selections[-1] == ([], "replace")


def test_shift_click_on_empty_space_keeps_the_selection(qtbot, label):
    """An accidental Shift+click on background must not wipe a hard-won
    multi-selection."""
    label.annotations = {"cell": [square(20, 20, 30)]}
    selections = _catch(label.canvasSelectionChanged)
    qtbot.mouseClick(label, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ShiftModifier, QPoint(180, 180))
    assert selections == []


# --- handle resize (#40) ---------------------------------------------------


def test_corner_handle_resize_anchors_the_opposite_corner(qtbot, label):
    annotation = square(50, 50, 50)          # bounds (50, 50) -> (100, 100)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]
    committed = _catch(label.bboxEditCommitted)

    _press_drag_release(qtbot, label, (100, 100), (150, 150))  # drag "br"

    assert committed == [True]
    xs = annotation["segmentation"][0::2]
    ys = annotation["segmentation"][1::2]
    assert min(xs) == pytest.approx(50), "anchored corner stays put"
    assert min(ys) == pytest.approx(50)
    assert max(xs) == pytest.approx(150), "dragged corner follows the mouse"
    assert max(ys) == pytest.approx(150)


def test_edge_handle_resizes_one_axis_only(qtbot, label):
    annotation = square(50, 50, 50)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]

    _press_drag_release(qtbot, label, (100, 75), (140, 75))  # drag "mr"

    ys = annotation["segmentation"][1::2]
    assert min(ys) == pytest.approx(50) and max(ys) == pytest.approx(100)
    assert max(annotation["segmentation"][0::2]) == pytest.approx(140)


def test_interior_drag_moves_the_shape(qtbot, label):
    annotation = square(50, 50, 40)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]

    _press_drag_release(qtbot, label, (70, 70), (100, 110))

    xs = annotation["segmentation"][0::2]
    ys = annotation["segmentation"][1::2]
    assert min(xs) == pytest.approx(80) and min(ys) == pytest.approx(90)
    assert max(xs) - min(xs) == pytest.approx(40), "a move must not resize"


def test_interior_click_without_drag_falls_through_to_selection(qtbot, label):
    """The move is drag-gated so a plain click can still pick a nested mask."""
    outer = square(20, 20, 100)
    inner = square(50, 50, 20, number=2)
    label.annotations = {"cell": [outer, inner]}
    label.highlighted_annotations = [outer]
    selections = _catch(label.canvasSelectionChanged)
    committed = _catch(label.bboxEditCommitted)

    qtbot.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))

    assert committed == [], "no drag happened, so nothing was edited"
    assert selections[-1] == ([inner], "replace"), "smallest containing mask wins"


def test_resize_is_unreachable_with_a_multi_selection(qtbot, label):
    """Handles are only draggable for exactly one selected shape."""
    first = square(50, 50, 40)
    second = square(120, 120, 40, number=2)
    label.annotations = {"cell": [first, second]}
    label.highlighted_annotations = [first, second]
    committed = _catch(label.bboxEditCommitted)

    _press_drag_release(qtbot, label, (90, 90), (140, 140))

    assert committed == []


def test_resize_clamps_into_the_image(qtbot, label):
    annotation = square(50, 50, 50)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]

    _press_drag_release(qtbot, label, (100, 100), (400, 400))

    assert max(annotation["segmentation"][0::2]) <= 200
    assert max(annotation["segmentation"][1::2]) <= 200


# --- deferred-gesture undo semantics (ADR-026) -----------------------------


def test_drag_requests_the_undo_baseline_at_gesture_start(qtbot, label):
    annotation = square(50, 50, 40)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]
    baselines = _catch(label.editBaselineRequested)

    qtbot.mousePress(label, Qt.MouseButton.LeftButton, pos=QPoint(90, 90))

    assert baselines == [True], "baseline captured on press, before any mutation"


def test_escape_during_a_drag_reverts_and_commits_nothing(qtbot, label):
    annotation = square(50, 50, 40)
    original = list(annotation["segmentation"])
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]
    committed = _catch(label.bboxEditCommitted)

    qtbot.mousePress(label, Qt.MouseButton.LeftButton, pos=QPoint(90, 90))
    qtbot.mouseMove(label, QPoint(140, 140))
    assert annotation["segmentation"] != original, "drag is live"
    qtbot.keyClick(label, Qt.Key.Key_Escape)

    assert annotation["segmentation"] == original
    assert committed == [], "an aborted gesture must leave no history entry"


# --- double-click into vertex-edit mode ------------------------------------


def test_double_click_enters_vertex_edit_mode(qtbot, label):
    annotation = square(40, 40, 60)
    label.annotations = {"cell": [annotation]}
    selected = _catch(label.annotationSelected)

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(70, 70))

    assert label.editing_polygon is annotation
    assert label._editing_polygon_orig == annotation["segmentation"]
    assert selected == [(annotation,)]


def test_double_click_picks_the_smallest_containing_polygon(qtbot, label):
    outer = square(10, 10, 150)
    inner = square(60, 60, 30, number=2)
    label.annotations = {"cell": [outer, inner]}

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(70, 70))

    assert label.editing_polygon is inner


def test_escape_leaves_vertex_edit_mode_reverting_drags(qtbot, label):
    annotation = square(40, 40, 60)
    original = list(annotation["segmentation"])
    label.annotations = {"cell": [annotation]}

    qtbot.mouseDClick(label, Qt.MouseButton.LeftButton, pos=QPoint(70, 70))
    label.editing_polygon["segmentation"][0] = 5  # simulate a vertex drag
    qtbot.keyClick(label, Qt.Key.Key_Escape)

    assert label.editing_polygon is None
    assert annotation["segmentation"] == original


# --- keypoint placement through real events (ADR-029 both-buttons) ---------


def test_right_click_reaches_the_keypoint_tool_through_the_dispatch(qtbot):
    """The press dispatch is left-button-only apart from the sam_points and
    keypoint short-circuits; without that branch the occluded-point path is
    unreachable from a real event."""
    ctx = FakeCanvasContext(
        current_class="person",
        classes=("person",),
        schemas={"person": {"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]}},
    )
    label = make_label(qtbot, ctx=ctx)
    label.show()
    qtbot.waitExposed(label)
    label.offset_x = label.offset_y = 0
    label.current_tool = "keypoint"

    qtbot.mouseClick(label, Qt.MouseButton.RightButton, pos=QPoint(30, 30))

    assert label.current_keypoints == [(30, 30, 1)]
