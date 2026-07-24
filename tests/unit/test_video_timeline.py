"""Unit tests for the VideoTimeline widget (issue #48).

Widget-only: exercises ``widgets/video_timeline.py`` with no main window. The
critical invariants are (1) ``set_current_frame`` NEVER re-emits
``frameSelected`` (a feedback loop back into ``switch_slice``), (2) genuine user
interaction DOES emit it exactly once, and (3) the fps guard keeps MM:SS finite.
"""

import pytest

from digitalsreeni_image_annotator.widgets.video_timeline import VideoTimeline


@pytest.fixture
def timeline(qt_application):
    tl = VideoTimeline()
    yield tl
    tl.deleteLater()


def test_set_video_configures_slider_range(timeline):
    timeline.set_video(100, 25.0)
    assert timeline.slider.minimum() == 0
    assert timeline.slider.maximum() == 99


def test_set_video_clamp_does_not_emit(qt_application):
    # THE feedback-loop invariant: set_video reconfigures the range, and if a
    # smaller total clamps a non-zero slider value Qt fires valueChanged. The
    # _updating guard must swallow it, or reconfiguring the timeline on an
    # annotation mutation would spuriously emit frameSelected -> switch_slice
    # -> jump to frame 0.
    from digitalsreeni_image_annotator.widgets.video_timeline import VideoTimeline

    tl = VideoTimeline()
    tl.set_video(100, 25.0)
    tl.set_current_frame(80)          # slider now at 80
    emitted = []
    tl.frameSelected.connect(emitted.append)
    tl.set_video(10, 25.0)            # setMaximum(9) clamps 80 -> 9 (valueChanged)
    assert emitted == []              # guard swallowed the clamp emission
    tl.deleteLater()


def test_set_current_frame_does_not_emit(timeline):
    """Programmatic sync (from switch_slice) must NOT emit frameSelected —
    otherwise it re-enters switch_slice (feedback loop)."""
    timeline.set_video(100, 25.0)
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    timeline.set_current_frame(50)

    assert emitted == []
    assert timeline.slider.value() == 50


def test_user_slider_move_emits_once(timeline):
    """A user-driven slider change (not programmatic, not a live drag) emits
    frameSelected exactly once."""
    timeline.set_video(100, 25.0)
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    # _updating is False and the handle isn't held down → valueChanged is a
    # genuine user move.
    timeline.slider.setValue(50)

    assert emitted == [50]


def test_label_shows_mmss_at_frame(timeline):
    """Frame 50 @ 25 fps is 2.0 s → 00:02 appears in the position label."""
    timeline.set_video(100, 25.0)
    timeline.set_current_frame(50)
    assert "00:02" in timeline.label.text()


def test_set_annotated_frames_stores(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({0, 99})
    assert timeline.annotated_frames == {0, 99}


def test_set_video_resets_annotated_marks(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({3, 4})
    timeline.set_video(50, 30.0)  # reconfigure → marks reset
    assert timeline.annotated_frames == set()


def test_zero_fps_does_not_divide_by_zero(timeline):
    """A 0/NaN fps container must not ZeroDivisionError; fps is guarded to 30."""
    timeline.set_video(10, 0)
    timeline.set_current_frame(5)  # _fmt_time would divide by zero without guard
    assert timeline.label.text()  # non-empty, finite MM:SS


def test_clear_resets_and_emits_nothing(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({1, 2})
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    timeline.clear()

    assert emitted == []
    assert timeline.annotated_frames == set()
    assert timeline.slider.maximum() == 0
    assert timeline.label.text() == ""


# --- per-frame states (issue #51) -------------------------------------------

def test_set_frame_states_stores_map_and_marked_set(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_frame_states({1: "annotated", 5: "tracked", 9: "needs_review"})
    assert timeline.frame_states == {1: "annotated", 5: "tracked", 9: "needs_review"}
    # annotated_frames is the back-compat set of ALL marked indices (any state).
    assert timeline.annotated_frames == {1, 5, 9}


def test_set_annotated_frames_delegates_to_states(timeline):
    """Back-compat: the set-based API still works AND now records state."""
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({0, 99})
    assert timeline.annotated_frames == {0, 99}
    assert timeline.frame_states == {0: "annotated", 99: "annotated"}


def test_frame_state_runs_collapses_contiguous_same_state(timeline):
    timeline.set_video(100, 25.0)
    # 3,4,5 share a state → one run; 7 is a lone tracked run; 8,9 needs_review.
    timeline.set_frame_states({
        3: "annotated", 4: "annotated", 5: "annotated",
        7: "tracked",
        8: "needs_review", 9: "needs_review",
    })
    assert timeline.frame_state_runs() == [
        (3, 5, "annotated"),
        (7, 7, "tracked"),
        (8, 9, "needs_review"),
    ]


def test_frame_state_runs_breaks_on_state_change_within_contiguous(timeline):
    timeline.set_video(100, 25.0)
    # Consecutive indices but a state change at 5 → two separate runs.
    timeline.set_frame_states({4: "annotated", 5: "tracked", 6: "tracked"})
    assert timeline.frame_state_runs() == [
        (4, 4, "annotated"),
        (5, 6, "tracked"),
    ]


def test_frame_state_runs_drops_out_of_range(timeline):
    timeline.set_video(10, 25.0)  # valid indices 0..9
    timeline.set_frame_states({8: "annotated", 9: "annotated", 12: "tracked"})
    assert timeline.frame_state_runs() == [(8, 9, "annotated")]


def test_set_video_resets_frame_states(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_frame_states({3: "tracked"})
    timeline.set_video(50, 30.0)
    assert timeline.frame_states == {}
    assert timeline.annotated_frames == set()
