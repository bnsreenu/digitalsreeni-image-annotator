"""Integration tests for SAM 3 video object tracking (issue #51, ADR-038).

Drives a full offscreen ``ImageAnnotator`` with a real tiny video, but the SAM 3
model is entirely stubbed: ``window.sam3_utils.track`` returns a canned
``[(frame_idx, result)]`` list whose scores straddle the threshold. Proves the
controller's commit / review-routing / rollback logic and the per-frame vs.
whole-run undo granularity — none of it needs weights or a GPU.

Uses the shared ``make_test_video`` fixture (tests/conftest.py). Modal dialogs
are neutralised (an offscreen modal hangs the run).
"""

import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMessageBox

from digitalsreeni_image_annotator.core.video_handler import frame_key


SEED_FRAME = 3

SEED = {"segmentation": [2.0, 2.0, 12.0, 2.0, 12.0, 10.0],
        "category_id": 1, "category_name": "cell", "number": 1}


def _seg(offset):
    o = float(offset)
    return [o, o, o + 8, o, o + 8, o + 6]


def _canned():
    """Per-frame track results straddling a 0.5 threshold. Frame 3 is the seed
    (must be skipped — the source annotation already lives there)."""
    return [
        (1, {"segmentation": _seg(1), "score": 0.9}),    # high → tracked commit
        (2, {"segmentation": _seg(2), "score": 0.3}),    # low  → needs_review
        (3, {"segmentation": _seg(3), "score": 0.99}),   # SEED → skipped
        (4, {"segmentation": _seg(4), "score": 0.8}),    # high → tracked commit
        (5, None),                                        # absent → nothing
    ]


class _FakeProgress:
    """Non-modal stand-in for the QProgressDialog (never cancels)."""

    def setWindowModality(self, *a):
        pass

    def setMinimumDuration(self, *a):
        pass

    def setValue(self, *a):
        pass

    def wasCanceled(self):
        return False

    def close(self):
        pass


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    w.auto_save = lambda: None  # no project file → skip recovery-snapshot writes
    yield w
    w.deleteLater()


@pytest.fixture(autouse=True)
def no_native_dialogs(monkeypatch):
    """Hard safety net: no modal may open in an offscreen run (it hangs)."""
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))


def _load_video(window, make_test_video, tmp_path):
    """Write a video into <tmp>/images and load it (so a save never prompts to
    copy). Returns the video's frame count."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    path = make_test_video(images, name="clip.avi", frames=8)
    window.current_project_dir = str(tmp_path)
    window.current_project_file = str(tmp_path / "proj.iap")
    window.image_controller.add_images_to_list([path])
    return window.video_handlers["clip"].total_frames


def _seed_and_select(window):
    """Navigate to the seed frame, add the source segmentation, select it."""
    window.add_class("cell", QColor("#ff0000"))
    window.image_controller.switch_slice(window.slice_list.item(SEED_FRAME))
    assert window.current_slice == frame_key("clip", SEED_FRAME)
    window.image_label.annotations = {"cell": [dict(SEED)]}
    window.save_current_annotations()
    window.image_label.highlighted_annotations = [
        window.image_label.annotations["cell"][0]
    ]


def _run_tracking(window, monkeypatch, canned=None):
    """Stub the model + dialogs and run one tracking pass. Returns run_id."""
    c = window.tracking_controller
    window.sam3_utils.loaded = True
    window.sam3_utils.track = lambda *a, **k: (canned if canned is not None else _canned())
    monkeypatch.setattr(c, "_prompt_tracking_options", lambda: 0.5)
    monkeypatch.setattr(c, "_make_progress_dialog", lambda: _FakeProgress())
    # Don't navigate on the review offer — keep the seed frame current.
    monkeypatch.setattr(window.dino_controller, "_show_dino_batch_review", lambda: None)
    c.run_tracking()
    return c._last_run["run_id"]


@pytest.fixture
def tracked(window, make_test_video, tmp_path, monkeypatch):
    n = _load_video(window, make_test_video, tmp_path)
    assert n >= 6
    _seed_and_select(window)
    run_id = _run_tracking(window, monkeypatch)
    return window, run_id


# --- routing ----------------------------------------------------------------

def test_run_tracking_routes_by_score(tracked):
    window, run_id = tracked

    # High-score frames committed with source sam3-track + a shared run id.
    f1 = window.all_annotations[frame_key("clip", 1)]["cell"]
    f4 = window.all_annotations[frame_key("clip", 4)]["cell"]
    assert len(f1) == 1 and len(f4) == 1
    assert f1[0]["source"] == "sam3-track" and f4[0]["source"] == "sam3-track"
    assert f1[0]["track_run"] == run_id == f4[0]["track_run"]
    assert window.tracking_controller._last_run["frames"] == [
        frame_key("clip", 1), frame_key("clip", 4)
    ]

    # Low-score frame → dino_batch_results (source "sam3"), NOT committed.
    assert frame_key("clip", 2) in window.dino_batch_results
    assert window.dino_batch_results[frame_key("clip", 2)][0]["source"] == "sam3"
    assert frame_key("clip", 2) not in window.all_annotations

    # Absent frame → nothing.
    assert frame_key("clip", 5) not in window.all_annotations


def test_seed_frame_not_duplicated(tracked):
    window, _ = tracked
    seed_cells = window.all_annotations[frame_key("clip", SEED_FRAME)]["cell"]
    # Only the original source annotation — no tracked copy on the seed frame.
    assert len(seed_cells) == 1
    assert "track_run" not in seed_cells[0]


# --- undo granularity -------------------------------------------------------

def test_per_frame_undo_removes_only_that_frame(tracked):
    window, run_id = tracked

    # Navigate to a tracked frame and Ctrl+Z (per-key granularity).
    window.image_controller.switch_slice(window.slice_list.item(1))
    assert window.current_slice == frame_key("clip", 1)
    window.annotation_controller.undo()

    # Frame 1's tracked annotation is gone; frame 4's survives.
    assert frame_key("clip", 1) not in window.all_annotations
    assert window.all_annotations[frame_key("clip", 4)]["cell"][0]["track_run"] == run_id


def test_undo_last_track_removes_run_keeps_preexisting(tracked):
    window, run_id = tracked

    # A pre-existing (non-run) annotation on a tracked frame must survive.
    window.all_annotations[frame_key("clip", 4)]["cell"].append(
        {"segmentation": _seg(6), "category_id": 1,
         "category_name": "cell", "number": 2}
    )

    window.tracking_controller.undo_last_track()

    # Frame 1 held only the tracked mask → the frame key is gone entirely.
    assert frame_key("clip", 1) not in window.all_annotations
    # Frame 4 keeps the pre-existing one; the run-tagged one is gone.
    f4 = window.all_annotations[frame_key("clip", 4)]["cell"]
    assert len(f4) == 1
    assert f4[0].get("track_run") != run_id
    assert window.tracking_controller._last_run is None


def test_undo_last_track_noop_without_run(window):
    # No run yet → a friendly no-op, no crash.
    window.tracking_controller.undo_last_track()
    assert window.tracking_controller._last_run is None


# --- accepting an uncertain frame reuses the DINO pipeline -------------------

def test_accept_uncertain_frame_commits_and_clears(tracked):
    window, _ = tracked

    # Navigate to the uncertain frame — _refresh_dino_temp_for_current surfaces
    # its pending result as temp_annotations.
    window.image_controller.switch_slice(window.slice_list.item(2))
    assert window.current_slice == frame_key("clip", 2)
    assert len(window.image_label.temp_annotations) == 1

    window.dino_controller.accept_dino_results()

    assert window.all_annotations[frame_key("clip", 2)]["cell"]
    assert frame_key("clip", 2) not in window.dino_batch_results


# --- persistence ------------------------------------------------------------

def test_tracked_annotations_survive_save_reload(tracked):
    window, run_id = tracked

    window.project_controller.save_project(show_message=False)
    proj = window.current_project_file
    window.project_controller.open_specific_project(proj)

    f1 = window.all_annotations[frame_key("clip", 1)]["cell"]
    assert f1[0]["source"] == "sam3-track"
    assert f1[0]["track_run"] == run_id
    assert f1[0]["segmentation"] == _seg(1)


# --- timeline states derived from the run -----------------------------------

def test_timeline_states_reflect_tracked_and_review(tracked):
    window, _ = tracked
    states = window.image_controller.video_frame_states("clip")
    assert states[1] == "tracked"
    assert states[4] == "tracked"
    assert states[2] == "needs_review"
    assert states[SEED_FRAME] == "annotated"  # the plain source annotation
    assert 5 not in states


# --- gating -----------------------------------------------------------------

def test_can_track_false_without_video(window):
    window.sam3_utils.loaded = True
    assert window.can_track() is False  # no video loaded


def test_can_track_false_without_selection(window, make_test_video, tmp_path):
    _load_video(window, make_test_video, tmp_path)
    window.sam3_utils.loaded = True
    window.image_label.highlighted_annotations = []
    assert window.can_track() is False


def test_can_track_false_for_pose_instance(window, make_test_video, tmp_path):
    _load_video(window, make_test_video, tmp_path)
    window.sam3_utils.loaded = True
    # A pose instance has a bbox but NO segmentation → excluded (ADR-029).
    pose = {"keypoints": [3.0, 3.0, 2], "num_keypoints": 1,
            "bbox": [2.0, 2.0, 4.0, 4.0],
            "category_id": 1, "category_name": "cell", "number": 1}
    window.image_label.highlighted_annotations = [pose]
    assert window.can_track() is False


def test_can_track_true_with_video_and_segmentation(window, make_test_video, tmp_path):
    _load_video(window, make_test_video, tmp_path)
    window.add_class("cell", QColor("#ff0000"))
    window.image_controller.switch_slice(window.slice_list.item(1))
    window.image_label.annotations = {"cell": [dict(SEED)]}
    window.image_label.highlighted_annotations = [
        window.image_label.annotations["cell"][0]
    ]
    window.sam3_utils.loaded = True
    assert window.can_track() is True
