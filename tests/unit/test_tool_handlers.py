"""Per-tool handler unit tests (issue #77, ADR-019).

Every ``ToolHandler`` subclass driven directly -- press, move, release, Enter,
Escape -- against a model-less ``ImageLabel`` and a fake ``CanvasContext``.

The assertions target the **emitted signal and its payload**, not internal
bookkeeping: the signal is the handler's contract with the controllers
(ADR-018), while the temp fields are implementation detail that a later
refactor is allowed to move. The one exception is the paint/eraser mask, whose
*existence* gates ``check_unsaved_changes`` and is therefore part of the
contract too -- but the intermediate mask contents are not asserted.
"""

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from tests.canvas_fixtures import (
    FakeCanvasContext,
    FakeMouseEvent,
    RecordingPainter,
    make_label,
)


@pytest.fixture
def label(qtbot):
    return make_label(qtbot)


def _catch(signal):
    """Collect a signal's emissions into a list (payload-less signals record True)."""
    seen = []
    signal.connect(lambda *args: seen.append(args if args else True))
    return seen


# --- RectangleTool ---------------------------------------------------------


def test_rectangle_drag_emits_finish_on_release(label):
    tool = label._tools["rectangle"]
    finished = _catch(label.finishRectangleRequested)

    assert tool.on_mouse_press(FakeMouseEvent(), (10, 20)) is True
    tool.on_mouse_move(FakeMouseEvent(), (60, 80))
    assert label.current_rectangle == [10, 20, 60, 80]
    assert tool.on_mouse_release(FakeMouseEvent(), (60, 80)) is True

    assert finished == [True]
    assert label.drawing_rectangle is False


def test_rectangle_normalises_a_backwards_drag(label):
    """Dragging up-left must still yield [xmin, ymin, xmax, ymax]."""
    tool = label._tools["rectangle"]
    tool.on_mouse_press(FakeMouseEvent(), (90, 90))
    tool.on_mouse_move(FakeMouseEvent(), (30, 40))
    assert label.current_rectangle == [30, 40, 90, 90]


def test_rectangle_ignores_right_button(label):
    tool = label._tools["rectangle"]
    ev = FakeMouseEvent(button=Qt.MouseButton.RightButton)
    assert tool.on_mouse_press(ev, (10, 10)) is False
    assert label.drawing_rectangle is False


def test_rectangle_release_without_drag_emits_nothing(label):
    """A press with no move leaves current_rectangle None -> no commit."""
    tool = label._tools["rectangle"]
    finished = _catch(label.finishRectangleRequested)
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_release(FakeMouseEvent(), (10, 10))
    assert finished == []


def test_rectangle_discard_clears_state(label):
    tool = label._tools["rectangle"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_move(FakeMouseEvent(), (50, 50))
    assert tool.has_unsaved_state() is True
    tool.discard()
    assert tool.has_unsaved_state() is False
    assert label.current_rectangle is None


# --- PolygonTool -----------------------------------------------------------


def test_polygon_accumulates_vertices(label):
    tool = label._tools["polygon"]
    for point in [(10, 10), (50, 10), (50, 50)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    assert label.current_annotation == [(10, 10), (50, 10), (50, 50)]
    assert label.drawing_polygon is True


def test_polygon_enter_finishes_at_three_points(label):
    tool = label._tools["polygon"]
    finished = _catch(label.finishPolygonRequested)
    for point in [(10, 10), (50, 10), (50, 50)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    assert tool.on_enter() is True
    assert finished == [True]


def test_polygon_enter_is_inert_below_three_points(label):
    tool = label._tools["polygon"]
    finished = _catch(label.finishPolygonRequested)
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_press(FakeMouseEvent(), (50, 10))
    assert tool.on_enter() is False
    assert finished == []


def test_polygon_double_click_finishes(label):
    tool = label._tools["polygon"]
    finished = _catch(label.finishPolygonRequested)
    for point in [(10, 10), (50, 10), (50, 50)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    assert tool.on_double_click(FakeMouseEvent(), (50, 50)) is True
    assert finished == [True]


def test_polygon_double_click_falls_through_when_not_drawing(label):
    """Not consuming the double-click is what lets it open vertex-edit mode."""
    tool = label._tools["polygon"]
    assert tool.on_double_click(FakeMouseEvent(), (50, 50)) is False


def test_polygon_escape_discards(label):
    tool = label._tools["polygon"]
    for point in [(10, 10), (50, 10), (50, 50)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    assert tool.on_escape() is True
    assert label.current_annotation == []
    assert label.drawing_polygon is False


def test_polygon_move_tracks_rubber_line(label):
    tool = label._tools["polygon"]
    assert tool.on_mouse_move(FakeMouseEvent(), (5, 5)) is False  # nothing started
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    assert tool.on_mouse_move(FakeMouseEvent(), (33, 44)) is True
    assert label.temp_point == (33, 44)


# --- PaintBrushTool --------------------------------------------------------


def test_paint_stroke_commits_one_annotation_per_blob(label):
    tool = label._tools["paint_brush"]
    committed = _catch(label.annotationCommitted)
    batch = _catch(label.annotationsBatchSaved)

    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    tool.on_mouse_move(FakeMouseEvent(), (60, 50))
    tool.on_mouse_release(FakeMouseEvent(), (60, 50))
    assert tool.has_unsaved_state() is True

    assert tool.on_enter() is True
    assert len(committed) == 1
    annotation = committed[0][0]
    assert annotation["category_name"] == "cell"
    assert len(annotation["segmentation"]) >= 6
    assert batch == [True]
    assert tool.has_unsaved_state() is False


def test_paint_captures_undo_baseline_at_stroke_start(label):
    """Deferred gesture: the baseline is captured on press, not on commit (ADR-026)."""
    tool = label._tools["paint_brush"]
    baselines = _catch(label.editBaselineRequested)
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    assert baselines == [True]


def test_paint_escape_discards_without_committing(label):
    tool = label._tools["paint_brush"]
    committed = _catch(label.annotationCommitted)
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    assert tool.on_escape() is True
    assert committed == []
    assert label.temp_paint_mask is None


def test_paint_commit_without_a_class_is_a_no_op(qtbot):
    label = make_label(qtbot, ctx=FakeCanvasContext(current_class=None))
    tool = label._tools["paint_brush"]
    committed = _catch(label.annotationCommitted)
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    tool.commit()
    assert committed == []
    # The mask survives, so the stroke isn't silently lost -- the user can pick
    # a class and commit again.
    assert label.temp_paint_mask is not None


def test_paint_move_without_press_does_nothing(label):
    tool = label._tools["paint_brush"]
    assert tool.on_mouse_move(FakeMouseEvent(), (10, 10)) is False
    assert label.temp_paint_mask is None


# --- EraserTool ------------------------------------------------------------


def test_eraser_commit_cuts_the_polygon_and_emits_replacement(label):
    label.annotations = {"cell": [
        {"segmentation": [10, 10, 190, 10, 190, 190, 10, 190],
         "category_name": "cell", "number": 1},
    ]}
    tool = label._tools["eraser"]
    replaced = _catch(label.annotationsReplaced)

    tool.on_mouse_press(FakeMouseEvent(), (100, 100))
    tool.on_mouse_release(FakeMouseEvent(), (100, 100))
    assert tool.on_enter() is True

    assert len(replaced) == 1
    image_key, annotations = replaced[0]
    assert image_key == "img.png"
    # The eraser bite leaves the mask still one connected region, so the
    # annotation survives -- but reshaped, which is the point.
    assert annotations["cell"], "erasing a hole must not delete the annotation"
    assert annotations["cell"][0]["segmentation"] != [10, 10, 190, 10, 190, 190, 10, 190]


def test_eraser_leaves_non_polygon_annotations_untouched(label):
    """A pose instance has no segmentation to cut -- it must pass through whole."""
    instance = {"keypoints": [50, 50, 2], "num_keypoints": 1,
                "bbox": [44, 44, 12, 12], "category_name": "person", "number": 1}
    label.annotations = {"person": [instance]}
    tool = label._tools["eraser"]
    replaced = _catch(label.annotationsReplaced)

    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    tool.on_enter()

    assert replaced[0][1]["person"] == [instance]


def test_eraser_escape_discards(label):
    tool = label._tools["eraser"]
    replaced = _catch(label.annotationsReplaced)
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    assert tool.on_escape() is True
    assert replaced == []
    assert label.temp_eraser_mask is None


def test_eraser_enter_without_a_stroke_is_inert(label):
    assert label._tools["eraser"].on_enter() is False


def test_eraser_mask_matches_the_image_size(label):
    tool = label._tools["eraser"]
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))
    assert label.temp_eraser_mask.shape == (200, 200)
    assert label.temp_eraser_mask.dtype == np.uint8


# --- KeypointTool (both-buttons path, ADR-029) -----------------------------


@pytest.fixture
def pose_label(qtbot):
    ctx = FakeCanvasContext(
        current_class="person",
        classes=("person",),
        schemas={"person": {"names": ["nose", "l_eye", "r_eye"],
                            "skeleton": [[0, 1]], "flip_idx": [0, 2, 1]}},
    )
    label = make_label(qtbot, ctx=ctx)
    label.current_tool = "keypoint"
    return label


def test_keypoint_right_button_places_an_occluded_point(pose_label):
    """Right-click = occluded (v=1). Missing this path was the ADR-029 trap:
    the tool short-circuits the left-only press dispatch precisely so the
    right button reaches it."""
    tool = pose_label._tools["keypoint"]
    tool.on_mouse_press(FakeMouseEvent(button=Qt.MouseButton.RightButton), (10, 10))
    assert pose_label.current_keypoints == [(10, 10, 1)]


def test_keypoint_left_button_places_a_visible_point(pose_label):
    tool = pose_label._tools["keypoint"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    assert pose_label.current_keypoints == [(10, 10, 2)]


def test_keypoint_auto_finishes_at_k(pose_label):
    tool = pose_label._tools["keypoint"]
    finished = _catch(pose_label.finishKeypointsRequested)
    for point in [(10, 10), (20, 10), (30, 10)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    assert finished == [True]


def test_keypoint_backspace_removes_the_last_point(pose_label):
    tool = pose_label._tools["keypoint"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_press(FakeMouseEvent(), (20, 10))
    assert tool.on_backspace() is True
    assert pose_label.current_keypoints == [(10, 10, 2)]
    assert pose_label.keypoint_next_index == 1


def test_keypoint_escape_discards_the_partial_pose(pose_label):
    tool = pose_label._tools["keypoint"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    assert tool.on_escape() is True
    assert pose_label.current_keypoints == []
    assert pose_label.drawing_keypoints is False


# --- in-progress overlays --------------------------------------------------
#
# ``paint_overlay`` runs on every repaint while a tool has state, so an
# exception there takes the whole canvas down rather than failing quietly.
# These assert it draws the expected primitive and, for the mask tools, that
# it survives the numpy -> QImage handoff (a stride mistake there is a crash,
# not a wrong pixel).


def test_polygon_overlay_draws_the_in_progress_outline(label):
    tool = label._tools["polygon"]
    for point in [(10, 10), (50, 10), (50, 50)]:
        tool.on_mouse_press(FakeMouseEvent(), point)
    tool.on_mouse_move(FakeMouseEvent(), (30, 60))

    painter = RecordingPainter()
    tool.paint_overlay(painter)

    assert painter.count("drawPolyline") == 1
    assert painter.count("drawEllipse") == 3, "one handle per placed vertex"
    assert painter.count("drawLine") == 1, "rubber line to the cursor"


def test_polygon_overlay_is_silent_with_no_polygon(label):
    painter = RecordingPainter()
    label._tools["polygon"].paint_overlay(painter)
    assert painter.calls == []


def test_rectangle_overlay_draws_the_preview(label):
    tool = label._tools["rectangle"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_move(FakeMouseEvent(), (60, 80))

    painter = RecordingPainter()
    tool.paint_overlay(painter)
    assert painter.count("drawRect") == 1


def test_rectangle_overlay_is_silent_after_release(label):
    tool = label._tools["rectangle"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_move(FakeMouseEvent(), (60, 80))
    tool.on_mouse_release(FakeMouseEvent(), (60, 80))

    painter = RecordingPainter()
    tool.paint_overlay(painter)
    assert painter.calls == []


@pytest.mark.parametrize("tool_name", ["paint_brush", "eraser"])
def test_mask_overlay_blits_the_stroke(label, tool_name):
    tool = label._tools[tool_name]
    tool.on_mouse_press(FakeMouseEvent(), (50, 50))

    painter = RecordingPainter()
    tool.paint_overlay(painter)

    assert painter.count("drawPixmap") == 1
    # Opacity is raised back to 1.0 afterwards; leaving it at 0.5 would fade
    # every layer painted after this one.
    opacities = [args[0] for name, args in painter.calls if name == "setOpacity"]
    assert opacities[-1] == 1.0


@pytest.mark.parametrize("tool_name", ["paint_brush", "eraser"])
def test_mask_overlay_is_silent_without_a_mask(label, tool_name):
    painter = RecordingPainter()
    label._tools[tool_name].paint_overlay(painter)
    assert painter.calls == []


def test_keypoint_overlay_draws_placed_points(pose_label):
    tool = pose_label._tools["keypoint"]
    tool.on_mouse_press(FakeMouseEvent(), (10, 10))
    tool.on_mouse_press(FakeMouseEvent(button=Qt.MouseButton.RightButton), (30, 10))

    painter = RecordingPainter()
    tool.paint_overlay(painter)

    assert painter.count("drawEllipse") == 2
    assert painter.count("drawLine") == 1, "schema skeleton edge between 0 and 1"
