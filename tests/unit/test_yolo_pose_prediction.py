"""Unit tests for YOLOController.process_yolo_results' pose branch (#35 PR-3).

Covers the new pose-task path (builds keypoint temp annotations, no
"segmentation" key, always v=2, seeds mw.keypoint_schemas) and confirms the
pre-existing segment-task path is unchanged. Uses a lightweight stub main
window instead of a real ImageAnnotator/QApplication window — the function
under test only touches a handful of ``mw`` attributes.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QObject

from src.digitalsreeni_image_annotator.controllers.yolo_controller import (
    YOLOController,
)


class _FakeTensor:
    """Stand-in for a torch.Tensor: only .cpu().numpy() is ever called."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = _FakeTensor(np.array([xyxy], dtype=float))


class _FakeKeypoints:
    def __init__(self, xy):
        # shape (1, K, 2) so `.xy.cpu().numpy()[0]` -> (K, 2)
        self.xy = _FakeTensor(np.array([xy], dtype=float))


class _FakeMask:
    def __init__(self, mask_2d):
        # shape (1, H, W) so `.data.cpu().numpy()[0]` -> (H, W)
        self.data = _FakeTensor(np.array([mask_2d], dtype=np.float32))


class _StubImageLabel:
    def __init__(self):
        self.annotations = {}
        self.class_colors = {}

    def update(self):
        pass


class _StubYoloTrainer:
    def __init__(self, task, class_names, prediction_keypoint_schema=None):
        self.model = SimpleNamespace(task=task)
        self.class_names = class_names
        self.prediction_keypoint_schema = prediction_keypoint_schema


class _StubMainWindow(QObject):
    """Subclasses QObject only because YOLOController.__init__ passes the
    main window through as its QObject parent -- process_yolo_results itself
    doesn't need Qt machinery on mw at all."""

    def __init__(self, image_paths, yolo_trainer):
        super().__init__()
        self.image_paths = image_paths
        self.keypoint_schemas = {}
        self.image_label = _StubImageLabel()
        self.yolo_trainer = yolo_trainer
        self.added_temp_classes = None

    def add_temp_classes(self, temp_annotations):
        self.added_temp_classes = temp_annotations

    def update_class_list(self):
        pass

    def deactivate_sam_tools(self):
        pass


@pytest.fixture(autouse=True)
def _silence_message_boxes(monkeypatch):
    """process_yolo_results ends with a modal QMessageBox.information() summary
    (and warns on error paths) -- under offscreen Qt these would still block
    waiting to be dismissed, so stub them out for every test in this file."""
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))


def _write_test_image(tmp_path, width=20, height=16):
    path = tmp_path / "test_image.png"
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(path)
    return str(path), width, height


def test_pose_branch_builds_keypoint_temp_annotations(tmp_path, qapp):
    image_path, width, height = _write_test_image(tmp_path)

    schema = {"names": ["kp0", "kp1"], "skeleton": [], "flip_idx": [0, 1]}
    trainer = _StubYoloTrainer(
        task="pose", class_names=["person"], prediction_keypoint_schema=schema
    )
    mw = _StubMainWindow({"test_image.png": image_path}, trainer)

    orig_shape = (height, width)  # (H, W) -- matches the real image so scale == 1.0
    kpts_xy = [[5.0, 6.0], [10.0, 11.0]]  # K=2 points
    box = _FakeBox(cls=0, conf=0.9, xyxy=[1.0, 2.0, 15.0, 14.0])
    fake_result = SimpleNamespace(
        boxes=[box],
        keypoints=[_FakeKeypoints(kpts_xy)],
        masks=None,
        orig_shape=orig_shape,
        orig_img=SimpleNamespace(shape=(height, width, 3)),
    )

    fake_results = ([fake_result], orig_shape, orig_shape)

    controller = YOLOController(mw)
    controller.process_yolo_results(fake_results, "test_image.png")

    assert mw.added_temp_classes is not None
    assert "Temp-person" in mw.added_temp_classes
    anns = mw.added_temp_classes["Temp-person"]
    assert len(anns) == 1
    ann = anns[0]

    assert "keypoints" in ann
    assert "bbox" in ann
    assert "segmentation" not in ann

    flat = ann["keypoints"]
    assert len(flat) == 2 * 3  # K=2 points * (x, y, v)
    visibilities = flat[2::3]
    assert visibilities == [2, 2]

    # scale == 1.0 (orig_shape matches the real image), so coords pass through.
    assert flat[0] == pytest.approx(5.0)
    assert flat[1] == pytest.approx(6.0)
    assert flat[3] == pytest.approx(10.0)
    assert flat[4] == pytest.approx(11.0)

    assert mw.keypoint_schemas.get("Temp-person") == schema
    assert mw.keypoint_schemas["Temp-person"] is not schema  # deep-copied, not aliased


def test_segment_branch_unchanged(tmp_path, qapp):
    image_path, width, height = _write_test_image(tmp_path)

    trainer = _StubYoloTrainer(task="segment", class_names=["cell"])
    mw = _StubMainWindow({"test_image.png": image_path}, trainer)

    orig_shape = (height, width)
    box = _FakeBox(cls=0, conf=0.85, xyxy=[1.0, 1.0, 10.0, 10.0])

    # A simple filled-square mask so cv2.findContours returns a real contour.
    mask_2d = np.zeros((height, width), dtype=np.float32)
    mask_2d[4:12, 4:12] = 1.0

    fake_result = SimpleNamespace(
        boxes=[box],
        keypoints=None,
        masks=[_FakeMask(mask_2d)],
        orig_shape=orig_shape,
        orig_img=SimpleNamespace(shape=(height, width, 3)),
    )

    fake_results = ([fake_result], orig_shape, orig_shape)

    controller = YOLOController(mw)
    controller.process_yolo_results(fake_results, "test_image.png")

    assert mw.added_temp_classes is not None
    assert "Temp-cell" in mw.added_temp_classes
    anns = mw.added_temp_classes["Temp-cell"]
    assert len(anns) == 1
    ann = anns[0]

    assert "segmentation" in ann
    assert "keypoints" not in ann
    assert len(ann["segmentation"]) > 0

    # Pose-only bookkeeping must not have been touched by the segment path.
    assert mw.keypoint_schemas == {}
