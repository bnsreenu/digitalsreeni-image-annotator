"""Integration tests for the video-timeline wiring + frame export (issue #48).

Drives a full offscreen ``ImageAnnotator``: loading a video shows the timeline,
a user scrub routes through ``switch_slice``, annotating a frame lights its mark,
switching to a plain image hides the timeline, and the Tools-menu frame export
writes exactly the annotated frames.

Uses the shared ``make_test_video`` fixture (tests/conftest.py). Modal dialogs
are neutralised by ``no_native_dialogs`` (an offscreen modal hangs the run).
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from digitalsreeni_image_annotator.core.video_handler import frame_key


POLY = {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0], "area": 31.5,
        "category_id": 1, "category_name": "cell", "number": 1}


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    w.auto_save = lambda: None  # no project file → skip recovery-snapshot writes
    yield w
    w.deleteLater()


@pytest.fixture
def no_native_dialogs(monkeypatch):
    """Hard safety net: no modal may open in an offscreen run (it hangs)."""
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))


def _load_video(window, make_test_video, tmp_path):
    path = make_test_video(tmp_path, name="clip.avi", frames=8)
    window.image_controller.add_images_to_list([path])
    return window.video_handlers["clip"].total_frames


def test_timeline_shown_for_video(window, make_test_video, tmp_path, no_native_dialogs):
    _load_video(window, make_test_video, tmp_path)
    window.update_video_timeline()
    # Under offscreen isVisible() is unreliable; isHidden() reflects the
    # explicit setVisible(True) the wiring performs.
    assert not window.video_timeline.isHidden()


def test_timeline_frame_selected_routes_through_switch_slice(
    window, make_test_video, tmp_path, no_native_dialogs
):
    n = _load_video(window, make_test_video, tmp_path)
    assert n >= 4

    # A user scrub emits frameSelected → on_timeline_frame_selected → switch_slice.
    window.video_timeline.frameSelected.emit(3)
    assert window.current_slice.endswith("_F00003")


def test_annotating_a_frame_lights_its_mark(
    window, make_test_video, tmp_path, no_native_dialogs
):
    n = _load_video(window, make_test_video, tmp_path)
    assert n >= 4
    window.add_class("cell", QColor("#ff0000"))

    # Annotate frame 2 through the normal save path.
    window.image_controller.switch_slice(window.slice_list.item(2))
    window.image_label.annotations = {"cell": [dict(POLY)]}
    window.save_current_annotations()  # routes through update_slice_list_colors

    assert 2 in window.image_controller.annotated_frame_indices("clip")
    # save_current_annotations refreshed the timeline via the mark choke point.
    assert 2 in window.video_timeline.annotated_frames


def test_annotation_refresh_does_not_emit_frame_selected(
    window, make_test_video, tmp_path, no_native_dialogs
):
    """An annotation mutation refreshes the timeline (update_slice_list_colors ->
    update_video_timeline -> set_video + set_current_frame) but must NOT emit
    frameSelected, which would re-enter switch_slice and jump the frame."""
    n = _load_video(window, make_test_video, tmp_path)
    assert n >= 4
    window.add_class("cell", QColor("#ff0000"))
    window.image_controller.switch_slice(window.slice_list.item(3))
    assert window.current_slice.endswith("_F00003")

    emitted = []
    window.video_timeline.frameSelected.connect(emitted.append)
    # Real mutation path: annotate + save routes through update_slice_list_colors,
    # which is the timeline-refresh hook.
    window.image_label.annotations = {"cell": [dict(POLY)]}
    window.save_current_annotations()

    assert emitted == []                             # no feedback loop
    assert window.current_slice.endswith("_F00003")  # frame unchanged
    assert 3 in window.video_timeline.annotated_frames


def test_timeline_hidden_for_plain_image(
    window, make_test_video, tmp_path, no_native_dialogs
):
    _load_video(window, make_test_video, tmp_path)

    png = tmp_path / "plain.png"
    plain = QImage(6, 5, QImage.Format.Format_RGB888)
    plain.fill(QColor("darkGreen"))
    plain.save(str(png))
    window.image_controller.add_images_to_list([str(png)])

    # Switch to the plain image → timeline hides.
    items = window.image_list.findItems("plain.png", Qt.MatchFlag.MatchExactly)
    assert items
    window.image_controller.switch_image(items[0])
    assert window.video_timeline.isHidden()


def test_export_annotated_frames_writes_only_annotated(
    window, make_test_video, tmp_path, no_native_dialogs, monkeypatch
):
    n = _load_video(window, make_test_video, tmp_path)
    assert n >= 6
    window.add_class("cell", QColor("#ff0000"))

    def annotate(idx):
        window.image_controller.switch_slice(window.slice_list.item(idx))
        window.image_label.annotations = {"cell": [dict(POLY)]}
        window.save_current_annotations()

    annotate(2)
    annotate(5)

    out_dir = tmp_path / "frames_out"
    out_dir.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(out_dir))
    )

    window.export_annotated_frames()

    written = sorted(p.name for p in out_dir.glob("*.png"))
    assert written == [f"{frame_key('clip', 2)}.png", f"{frame_key('clip', 5)}.png"]
    for name in written:
        img = QImage(str(out_dir / name))
        assert not img.isNull()
