"""Unit tests for the KeypointTool placement logic (issue #35).

Drives the handler on a real (but model-less) ImageLabel with a fake canvas
context supplying a 3-point schema. Verifies ordered placement, visibility by
button/modifier, auto-finish at K, finish-early (Enter), Backspace go-back, and
Esc discard — without a QApplication event loop.
"""

import pytest
from PyQt6.QtCore import Qt

from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel


class _FakeCtx:
    def __init__(self, schema):
        self._schema = schema

    def current_class(self):
        return "person"

    def keypoint_schema(self, name):
        return self._schema if name == "person" else None


class _FakeEvent:
    def __init__(self, button=Qt.MouseButton.LeftButton, shift=False):
        self._button = button
        self._shift = shift

    def button(self):
        return self._button

    def modifiers(self):
        return (
            Qt.KeyboardModifier.ShiftModifier
            if self._shift
            else Qt.KeyboardModifier.NoModifier
        )


@pytest.fixture
def tool(qtbot):
    label = ImageLabel(None)
    qtbot.addWidget(label)
    label.zoom_factor = 1.0
    label.ui_scale = 1.0
    label._ctx = _FakeCtx(
        {"names": ["nose", "l_eye", "r_eye"], "skeleton": [[0, 1]], "flip_idx": [0, 2, 1]}
    )
    label.current_tool = "keypoint"
    return label._tools["keypoint"]


def _finishes(label):
    fired = []
    label.finishKeypointsRequested.connect(lambda: fired.append(True))
    return fired


def test_left_click_places_visible_points(tool):
    label = tool.label
    tool.on_mouse_press(_FakeEvent(), (10, 20))
    assert label.drawing_keypoints
    assert label.current_keypoints == [(10, 20, 2)]
    assert label.keypoint_next_index == 1


def test_right_click_and_shift_place_occluded(tool):
    label = tool.label
    tool.on_mouse_press(_FakeEvent(button=Qt.MouseButton.RightButton), (5, 5))
    tool.on_mouse_press(_FakeEvent(shift=True), (6, 6))
    assert [v for _, _, v in label.current_keypoints] == [1, 1]


def test_auto_finish_at_k(tool):
    label = tool.label
    fired = _finishes(label)
    for i in range(3):  # K == 3
        tool.on_mouse_press(_FakeEvent(), (i, i))
    assert fired == [True]
    assert len(label.current_keypoints) == 3


def test_enter_finishes_early(tool):
    label = tool.label
    fired = _finishes(label)
    tool.on_mouse_press(_FakeEvent(), (1, 1))
    assert tool.on_enter() is True
    assert fired == [True]


def test_enter_noop_without_points(tool):
    assert tool.on_enter() is False


def test_backspace_removes_last_point(tool):
    label = tool.label
    tool.on_mouse_press(_FakeEvent(), (1, 1))
    tool.on_mouse_press(_FakeEvent(), (2, 2))
    assert tool.on_backspace() is True
    assert label.current_keypoints == [(1, 1, 2)]
    assert label.keypoint_next_index == 1
    # Removing the last point clears the drawing flag.
    tool.on_backspace()
    assert not label.drawing_keypoints
    assert tool.on_backspace() is False


def test_escape_discards(tool):
    label = tool.label
    tool.on_mouse_press(_FakeEvent(), (1, 1))
    assert tool.on_escape() is True
    assert label.current_keypoints == [] and not label.drawing_keypoints


def test_no_schema_is_noop(qtbot):
    label = ImageLabel(None)
    qtbot.addWidget(label)
    label._ctx = _FakeCtx(None)  # current class has no schema
    handler = label._tools["keypoint"]
    assert handler.on_mouse_press(_FakeEvent(), (1, 1)) is False
    assert not label.drawing_keypoints


def test_has_unsaved_state_tracks_placement(tool):
    assert tool.has_unsaved_state() is False
    tool.on_mouse_press(_FakeEvent(), (1, 1))
    assert tool.has_unsaved_state() is True
