"""Training and navigating a video's frames (regressions found in manual use).

Two bugs, one theme: a video's frames are ordinary slices everywhere else in
the app, and two places forgot it.

1. ``multidimensional_blockers`` refused to train on any stack or video, citing
   a constraint that predated slice-aware export. Annotating 368 polygons
   across a video's frames produced a perfectly good dataset that the training
   dialog then declined to use.
2. The slice list never switched the canvas on Up/Down, because only
   ``itemClicked`` was connected and a focused ``QListWidget`` consumes the
   arrow keys itself.

These use a REAL video written by the ``make_test_video`` fixture and the real
``load_video`` path, because both bugs were in the wiring between the video
machinery and something that assumed a plain image.
"""

import os

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from digitalsreeni_image_annotator.dialogs.yolo_trainer import YOLOTrainer
from digitalsreeni_image_annotator.io.export_formats import export_yolo_v5plus


@pytest.fixture
def window(qt_application, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    monkeypatch.setattr(w, "auto_save", lambda *a, **k: None)
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    yield w
    w.deleteLater()


@pytest.fixture
def video_window(window, make_test_video, tmp_path):
    """A real video added through the real add path.

    ``add_images_to_list`` is what registers the ``is_video`` entry in
    ``all_images`` that the training dialog reads -- calling ``load_video``
    directly loads the frames but leaves the project with no images at all,
    which would make the dialog assertions vacuous.
    """
    path = make_test_video(tmp_path, frames=6)
    window.add_images_to_list([path])
    assert window.all_images, "video was not registered"
    return window, path


def _polygon():
    return {
        "segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0],
        "category_name": "bee",
        "category_id": 1,
    }


# --- arrow-key navigation --------------------------------------------------


def test_arrow_keys_move_the_canvas_through_the_frames(video_window, qtbot):
    window, _ = video_window
    first = window.current_slice
    assert window.slice_list.count() == 6

    window.slice_list.setFocus()
    qtbot.keyClick(window.slice_list, Qt.Key.Key_Down)

    assert window.current_slice != first, (
        "the row moved but the canvas did not follow it"
    )
    assert window.slice_list.currentItem().text() == window.current_slice
    assert window.current_image is not None


def test_arrow_keys_walk_further_than_one_frame(video_window, qtbot):
    window, _ = video_window
    window.slice_list.setFocus()
    for _ in range(3):
        qtbot.keyClick(window.slice_list, Qt.Key.Key_Down)

    assert window.slice_list.currentRow() == 3
    assert window.current_slice == window.slice_list.item(3).text()


def test_a_timeline_scrub_keeps_the_row_in_sync(video_window, qtbot):
    """Scrub to frame 4, press Down, land on frame 5 -- not frame 1.

    ``on_timeline_frame_selected`` used to switch the canvas without moving the
    row, and the arrow keys step from the row. Before the list drove navigation
    that desync was invisible (Down was inert); afterwards it teleports.
    """
    window, _ = video_window
    window.on_timeline_frame_selected(4)

    assert window.slice_list.currentRow() == 4
    assert window.current_slice == window.slice_list.item(4).text()

    window.slice_list.setFocus()
    qtbot.keyClick(window.slice_list, Qt.Key.Key_Down)

    assert window.slice_list.currentRow() == 5
    assert window.current_slice == window.slice_list.item(5).text()


def test_reselecting_the_current_frame_is_a_no_op(video_window):
    """The guard that lets currentRowChanged drive navigation without
    blockSignals at a dozen programmatic-selection call sites."""
    window, _ = video_window
    saves = []
    window.save_current_annotations = lambda *a, **k: saves.append(1)

    window.image_controller.switch_slice(window.slice_list.currentItem())

    assert saves == []


def test_switching_videos_does_not_bleed_annotations_across_them(
    window, make_test_video, tmp_path
):
    """The hazard the already-current guard actually prevents.

    ``update_slice_list`` re-selects the current slice *after* ``switch_image``
    has already repointed ``current_slice`` at the incoming video's first
    frame. Without the guard that re-selection re-enters ``switch_slice``,
    which saves the outgoing frame's annotations under the INCOMING frame's
    key -- the same polygon silently lands in two different videos.

    The previous test pokes the guard directly and so passes for the wrong
    reason; this one fails if the guard is deleted.
    """
    first = make_test_video(tmp_path, name="aaa.avi", frames=4)
    second = make_test_video(tmp_path, name="bbb.avi", frames=4)
    window.add_images_to_list([first, second])
    window.add_class("bee", QColor("#ffa500"))

    # Land on the first video and annotate a frame that is not its first.
    window.image_list.setCurrentRow(0)
    window.image_controller.switch_image(window.image_list.item(0))
    window.slice_list.setCurrentRow(2)
    annotated = window.current_slice
    assert annotated.startswith("aaa")
    window.image_label.annotations = {"bee": [_polygon()]}

    window.image_controller.switch_image(window.image_list.item(1))

    bled = [
        name
        for name, by_class in window.all_annotations.items()
        if by_class and name.startswith("bbb")
    ]
    assert bled == [], f"annotations leaked into the second video: {bled}"
    assert window.all_annotations.get(annotated, {}).get("bee")


# --- predicting on a video's frames ----------------------------------------


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _FakeBox:
    cls = 0
    conf = 0.9


class _FakeMask:
    """One filled rectangle, so the polygon that comes out has coordinates
    worth asserting on. Half the frame wide, a quarter of it tall."""

    def __init__(self, width, height):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[: height // 4, : width // 2] = 1.0
        self.data = _FakeTensor(mask[None, ...])


class _FakeResult:
    def __init__(self, width=None, height=None):
        if width is None:
            self.boxes, self.masks = [], None
        else:
            self.boxes = [_FakeBox()]
            self.masks = [_FakeMask(width, height)]
        self.keypoints = None


class _FakeTrainer:
    """Reports the size it was actually handed, rather than a hardcoded 640.

    That matters: process_yolo_results divides the size the caller passes by
    the size in the results, so a trainer that always claims 640x640 would let
    a (height, width) transposition through unnoticed.
    """

    def __init__(self, with_mask=False):
        self.model = object()
        self.class_names = ["bee"]
        self.prediction_keypoint_schema = None
        self.with_mask = with_mask
        self.sources = []

    def predict(self, source):
        self.sources.append(source)
        height, width = source.shape[:2]
        result = _FakeResult(width, height) if self.with_mask else _FakeResult()
        return [result], (height, width), (height, width)

    # The real accessor rather than a simplified stand-in, so the controller
    # call sites are exercised against the code under test.
    class_name_for = YOLOTrainer.class_name_for


def test_predicting_on_a_video_frame_does_not_crash(video_window, monkeypatch):
    """The crash: "Try it on the current image" straight after training on a
    video did ``image_paths[file_name]`` with a FRAME name and raised KeyError
    out of a Qt slot, taking the app down.

    A frame has no path; its pixels live in the lazy slice collection, so the
    frame must resolve to an array instead.
    """
    from PyQt6.QtWidgets import QMessageBox

    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    window.slice_list.setCurrentRow(2)
    frame = window.current_slice
    assert frame not in window.image_paths, "a frame is not a path key"

    trainer = _FakeTrainer()
    window.yolo_trainer = trainer
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window.yolo_controller.predict_single_image(frame)

    assert len(trainer.sources) == 1
    source = trainer.sources[0]
    assert not isinstance(source, str), "a frame must be predicted on as pixels"
    assert source.shape[2] == 3
    assert source.flags["C_CONTIGUOUS"], "torch refuses a negative-stride view"


@pytest.mark.parametrize("width", [32, 34])
def test_a_frame_predicts_at_its_own_dimensions(
    window, make_test_video, tmp_path, monkeypatch, width
):
    """The scaling half, which the resolution tests never touch.

    Width 34 is not a multiple of 4, so Qt pads each RGB888 scanline by two
    bytes. Reading rows on the width instead of the stride shears the image
    progressively -- and 32, like 640 and 1024, hides it completely.
    """
    from PyQt6.QtWidgets import QMessageBox

    path = make_test_video(tmp_path, name=f"w{width}.avi", frames=4, width=width,
                           height=24)
    window.add_images_to_list([path])
    window.add_class("bee", QColor("#ffa500"))
    window.slice_list.setCurrentRow(2)

    trainer = _FakeTrainer(with_mask=True)
    window.yolo_trainer = trainer
    captured = {}
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        window, "add_temp_classes", lambda temp: captured.update(temp)
    )

    window.yolo_controller.predict_single_image(window.current_slice)

    source = trainer.sources[0]
    assert source.shape[:2] == (24, width), "predicted at the wrong size"

    polygon = captured["Temp-bee"][0]["segmentation"]
    xs, ys = polygon[0::2], polygon[1::2]
    # The mask fills the left half and top quarter, and scale is 1:1 because
    # the trainer reports the size it was given.
    assert max(xs) == pytest.approx(width / 2, abs=2)
    assert max(ys) == pytest.approx(24 / 4, abs=2)

    # Uniform frame: every row must carry the same colour. A stride bug makes
    # the lower rows drift.
    assert (source[0] == source[-1]).all(), "scanline padding sheared the frame"


def test_a_frame_is_predicted_in_bgr(video_window, monkeypatch):
    """Ultralytics treats a numpy source as BGR and flips it internally.
    Handing it RGB produces plausible, quietly worse predictions rather than an
    error -- so the channel order is worth pinning. make_test_video ramps the
    RED channel, so BGR puts that ramp in index 2."""
    from PyQt6.QtWidgets import QMessageBox

    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    window.slice_list.setCurrentRow(3)

    trainer = _FakeTrainer()
    window.yolo_trainer = trainer
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window.yolo_controller.predict_single_image(window.current_slice)

    blue, green, red = trainer.sources[0][1, 1]
    assert red > blue and red > green, (
        f"channels look swapped: got B={blue} G={green} R={red}"
    )


def test_predicting_on_a_name_with_no_pixels_reports_rather_than_raises(
    video_window, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    window, _ = video_window
    window.yolo_trainer = _FakeTrainer()
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: warnings.append(" ".join(map(str, a)))),
    )

    window.yolo_controller.predict_single_image("nowhere_F00001")

    assert warnings and "nowhere_F00001" in warnings[0]


# --- training on a video's frames ------------------------------------------


def test_the_train_dialog_accepts_annotated_video_frames(video_window, qtbot):
    from digitalsreeni_image_annotator.dialogs.train_dialog import TrainDialog

    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frame = window.slice_list.item(1).text()
    window.all_annotations[frame] = {"bee": [_polygon()]}

    dialog = TrainDialog(window)
    qtbot.addWidget(dialog)

    assert dialog.train_button.isEnabled() is True, dialog.blocker_label.text()
    # The frames are the images, not the video file.
    assert "1 of 6 image(s)" in dialog.data_label.text()


def test_video_frames_are_written_by_the_yolo_export(video_window, tmp_path):
    """The export half. A frame resolves through the video's LazySliceList
    exactly like a stack slice does."""
    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frames = [window.slice_list.item(i).text() for i in (1, 3)]
    for frame in frames:
        window.all_annotations[frame] = {"bee": [_polygon()]}

    out = tmp_path / "dataset"
    export_yolo_v5plus(
        window.all_annotations,
        {"bee": 1},
        window.image_paths,
        window.slices,
        window.image_slices,
        str(out),
        val_split=0,
    )

    images = os.listdir(out / "images" / "train")
    labels = os.listdir(out / "labels" / "train")
    assert sorted(images) == sorted(f"{name}.png" for name in frames)
    assert sorted(labels) == sorted(f"{name}.txt" for name in frames)
    for label in labels:
        assert (out / "labels" / "train" / label).read_text().strip()

    # And they are the RIGHT frames. make_test_video ramps the red channel by
    # frame index precisely so a mis-resolution is detectable; asserting only
    # on filenames would pass while writing frame 0 four times.
    from PIL import Image

    from digitalsreeni_image_annotator.core.video_handler import parse_frame_index

    for name in frames:
        pixel = Image.open(out / "images" / "train" / f"{name}.png").convert("RGB")
        red = pixel.getpixel((1, 1))[0]
        assert abs(red - 10 * parse_frame_index(name)) <= 2, (
            f"{name} carries frame {red / 10:.0f}'s pixels"
        )


def test_a_frame_of_a_non_active_video_still_exports(
    window, make_test_video, tmp_path
):
    """The semantic widening in the export change: resolution now spans every
    collection in ``image_slices``, not just the active one. Annotate video A,
    switch to video B, export -- A's frame must still be written."""
    first = make_test_video(tmp_path, name="aaa.avi", frames=4)
    second = make_test_video(tmp_path, name="bbb.avi", frames=4)
    window.add_images_to_list([first, second])
    window.add_class("bee", QColor("#ffa500"))

    window.image_controller.switch_image(window.image_list.item(0))
    frame = window.slice_list.item(1).text()
    window.all_annotations[frame] = {"bee": [_polygon()]}
    window.image_controller.switch_image(window.image_list.item(1))
    assert window.slices is not window.image_slices["aaa"], "A is still active"

    out = tmp_path / "ds"
    export_yolo_v5plus(
        window.all_annotations, {"bee": 1}, window.image_paths,
        window.slices, window.image_slices, str(out), val_split=0,
    )

    assert os.listdir(out / "images" / "train") == [f"{frame}.png"]


def test_the_export_does_not_materialise_every_frame(video_window, tmp_path):
    """Building a ``{name: qimage}`` map over the whole collection decoded and
    held every frame at once, defeating the bounded lazy cache (#45). Only the
    annotated frames should ever be decoded."""
    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frame = window.slice_list.item(2).text()
    window.all_annotations[frame] = {"bee": [_polygon()]}

    decoded = []
    real_get = window.slices.get

    def counting_get(name):
        decoded.append(name)
        return real_get(name)

    window.slices.get = counting_get
    export_yolo_v5plus(
        window.all_annotations,
        {"bee": 1},
        window.image_paths,
        window.slices,
        window.image_slices,
        str(tmp_path / "ds"),
        val_split=0,
    )

    assert decoded == [frame], f"decoded {len(decoded)} frame(s), wanted 1"
