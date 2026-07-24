"""Integration tests for video-frames-as-slices (issue #47).

Drives a full offscreen ``ImageAnnotator`` through the real load / navigate /
save / reload paths, proving a video reuses the #45 lazy-slice machinery: its
``image_slices[base]`` is a ``LazySliceList`` and saving decodes no frames.

The video is built with the shared ``make_test_video`` fixture
(tests/conftest.py). Modal dialogs are neutralised by ``no_native_dialogs``
(an offscreen modal hangs the run).
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from digitalsreeni_image_annotator.core.slice_cache import LazySliceList
from digitalsreeni_image_annotator.core.video_handler import frame_key


POLY = {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0], "area": 31.5,
        "category_id": 1, "category_name": "cell", "number": 1}


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
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
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], ""))
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )


def _slice_list_texts(window):
    return [
        window.slice_list.item(i).text()
        for i in range(window.slice_list.count())
    ]


def test_load_video_creates_lazy_frame_slices(window, make_test_video, tmp_path):
    """load_video builds a LazySliceList of frame slices (the #45 object), the
    slice list is populated, and navigating to a far frame materialises it."""
    path = make_test_video(tmp_path, name="clip.avi", frames=8)

    window.image_controller.load_video(path)

    lazy = window.image_slices["clip"]
    n = window.video_handlers["clip"].total_frames
    assert isinstance(lazy, LazySliceList)
    assert window.slices is lazy          # same object (issue #45 guardrail)
    assert len(lazy) == n
    assert lazy.names == [frame_key("clip", i) for i in range(n)]
    assert _slice_list_texts(window) == lazy.names

    # First frame is current + materialised after load.
    assert window.current_slice == frame_key("clip", 0)
    assert isinstance(window.current_image, QImage)

    # Navigate to frame 3 — decoded lazily on demand.
    target = frame_key("clip", 3)
    items = window.slice_list.findItems(target, Qt.MatchFlag.MatchExactly)
    assert items
    window.image_controller.switch_slice(items[0])
    assert window.current_slice == target
    assert isinstance(window.current_image, QImage)
    assert not window.current_image.isNull()


def test_base_name_collision_refused(
    window, make_test_video, tmp_path, no_native_dialogs
):
    """A second file whose ext-stripped base collides with a loaded video is
    refused (slices/annotations are keyed by the ext-stripped base)."""
    video = make_test_video(tmp_path, name="clip.avi", frames=8)
    window.image_controller.add_images_to_list([video])
    assert "clip" in window.image_slices

    # A PNG named "clip.png" shares the base name "clip" → refused.
    png = tmp_path / "clip.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(png))
    window.image_controller.add_images_to_list([str(png)])

    assert "clip.png" not in window.image_paths
    assert not any(i["file_name"] == "clip.png" for i in window.all_images)


def test_switch_between_video_and_image_reuses_lazy_slices(
    window, make_test_video, tmp_path, no_native_dialogs
):
    """switch_image between a video and a plain image and back exercises the
    `base in image_slices` reuse branch the #45/#47 reconciliation rests on:
    the video's LazySliceList is restored intact (no rebuild, no stale
    display)."""
    video = make_test_video(tmp_path, name="clip.avi", frames=8)
    png = tmp_path / "plain.png"
    plain = QImage(6, 5, QImage.Format.Format_RGB888)
    plain.fill(Qt.GlobalColor.darkGreen)
    plain.save(str(png))

    window.image_controller.add_images_to_list([video, str(png)])
    lazy = window.image_slices["clip"]

    def item(name):
        its = window.image_list.findItems(name, Qt.MatchFlag.MatchExactly)
        assert its, name
        return its[0]

    # Switch to the plain image: slice list cleared, a single 2D image shown.
    window.image_controller.switch_image(item("plain.png"))
    assert window.current_slice is None
    assert window.slice_list.count() == 0
    assert isinstance(window.current_image, QImage)

    # Switch back to the video: the SAME LazySliceList is restored via the
    # reuse branch (not rebuilt), frame 0 current, slice list repopulated.
    window.image_controller.switch_image(item("clip.avi"))
    assert window.image_slices["clip"] is lazy
    assert window.slices is lazy
    assert window.current_slice == frame_key("clip", 0)
    assert window.slice_list.count() == len(lazy)


def test_video_roundtrip_save_reload(
    window, make_test_video, tmp_path, no_native_dialogs
):
    """Open a video, annotate a far frame, save, reload: the annotation is
    restored under the frame key, the reloaded entry carries is_video /
    video_metadata, and saving decoded NO frames (lazy holds)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    path = make_test_video(images_dir, name="clip.avi", frames=8)

    window.current_project_file = str(tmp_path / "proj.iap")
    window.current_project_dir = str(tmp_path)
    window.add_class("cell", QColor("#ff0000"))

    window.image_controller.add_images_to_list([path])
    handler = window.video_handlers["clip"]
    target = frame_key("clip", 3)

    items = window.slice_list.findItems(target, Qt.MatchFlag.MatchExactly)
    assert items
    window.image_controller.switch_slice(items[0])
    assert window.current_slice == target

    # Annotate the (far) current frame through the normal save path.
    window.image_label.annotations = {"cell": [dict(POLY)]}
    window.save_current_annotations()

    # Spy on decode AFTER annotating — the save must touch no pixels.
    calls = {"n": 0}
    original = handler.get_frame

    def counting(idx):
        calls["n"] += 1
        return original(idx)

    handler.get_frame = counting

    window.project_controller.save_project(show_message=False)
    assert calls["n"] == 0  # placeholders / lazy holds — save decoded nothing

    # Reload from disk.
    window.project_controller.open_specific_project(
        str(window.current_project_file)
    )

    # Annotation restored under the frame key.
    assert target in window.all_annotations
    assert (
        window.all_annotations[target]["cell"][0]["segmentation"]
        == POLY["segmentation"]
    )

    # Reloaded image entry is still a video with its metadata.
    reloaded = next(
        i for i in window.all_images if i["file_name"] == "clip.avi"
    )
    assert reloaded.get("is_video") is True
    assert reloaded.get("video_metadata")
    assert reloaded["video_metadata"]["total_frames"] == handler.total_frames
    # Reload rebuilt the lazy frame slices.
    assert isinstance(window.image_slices["clip"], LazySliceList)


@pytest.mark.parametrize("remove_method", ["remove_image", "delete_selected_image"])
def test_remove_video_clears_orphaned_slices(
    window, make_test_video, tmp_path, no_native_dialogs, remove_method
):
    """Removing a video (via the context-menu Remove path OR the Delete-key
    ``delete_selected_image`` path) must drop its frame slices AND reset
    ``mw.slices``.

    Regression: both methods deleted ``image_slices[base]`` + released the
    handler + cleared the slice-list widget, but left ``mw.slices`` pointing at
    the removed video's LazySliceList. ``update_ui`` -> ``update_slice_list``
    then rebuilt the list from that dangling ref, so the orphaned frames
    reappeared and selecting one showed no preview (its handler was released).
    """
    video = make_test_video(tmp_path, name="clip.avi", frames=8)
    window.image_controller.add_images_to_list([video])
    assert "clip" in window.image_slices
    assert window.slice_list.count() == 8

    window.image_list.setCurrentRow(0)          # select the (only) video
    getattr(window.image_controller, remove_method)()

    assert "clip" not in window.image_slices
    assert "clip" not in window.video_handlers
    assert window.slices == []                  # dangling ref dropped
    assert window.slice_list.count() == 0
    # A UI refresh must NOT resurrect the orphaned frames from mw.slices.
    window.update_ui()
    assert window.slice_list.count() == 0


def test_add_images_button_path_accepts_and_loads_video(
    window, make_test_video, tmp_path, monkeypatch
):
    """The visible "Add Images / Videos" button (``add_images``) must let the user
    pick AND load a video.

    Regression for the #47/#48 gap: ``add_images`` — the only method the
    sidebar button is wired to — kept an image-only file filter
    (``*.png…*.czi``), so no ``.mp4/.avi/.mov`` could be selected even though
    the downstream ``add_images_to_list`` fully supports video. The earlier
    video tests all called ``add_images_to_list``/``load_video`` directly and
    so drove *below* the dialog, missing this. This drives the full button
    path: the dialog's filter must offer the video extensions, and a selected
    video must load as lazy frame slices.
    """
    path = make_test_video(tmp_path, name="clip.avi", frames=8)

    captured = {}

    def fake_get_open(parent, caption, directory, filt):
        captured["filter"] = filt
        return ([path], "")

    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", staticmethod(fake_get_open)
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))

    window.add_images()

    # The dialog must offer the video container extensions...
    filt = captured["filter"].lower()
    assert ".mp4" in filt and ".avi" in filt and ".mov" in filt
    # ...and the selected video must actually load as lazy frame slices.
    assert isinstance(window.image_slices.get("clip"), LazySliceList)
