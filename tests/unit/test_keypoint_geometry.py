"""Unit tests for ImageLabel's pure keypoint geometry helpers (issue #35).

``_keypoint_bounds`` / ``_annotation_bbox`` / ``_annotation_contains`` keypoint
fallbacks and the affine ``_scale_keypoints`` / ``_translate_keypoints`` used by
the instance-box transform. Static methods — no widget instance needed.
"""

from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel


def test_keypoint_bounds_over_labelled_points_only():
    # v=0 padded point at (0,0) must not pull the bounds to the origin.
    ann = {"keypoints": [10, 20, 2, 40, 60, 1, 0, 0, 0]}
    assert ImageLabel._keypoint_bounds(ann) == (10, 20, 40, 60)


def test_keypoint_bounds_none_when_nothing_labelled():
    assert ImageLabel._keypoint_bounds({"keypoints": [0, 0, 0]}) is None
    assert ImageLabel._keypoint_bounds({"keypoints": []}) is None
    assert ImageLabel._keypoint_bounds({}) is None


def test_annotation_bbox_prefers_stored_box_over_points():
    # The stored bbox is authoritative (it's the resizable instance box).
    ann = {"keypoints": [10, 20, 2, 40, 60, 2], "bbox": [0, 0, 100, 100]}
    assert ImageLabel._annotation_bbox(ann) == (0, 0, 100, 100)


def test_annotation_bbox_falls_back_to_points_without_box():
    ann = {"keypoints": [10, 20, 2, 40, 60, 2]}
    assert ImageLabel._annotation_bbox(ann) == (10, 20, 40, 60)


def test_annotation_contains_uses_keypoint_bounds_without_box():
    ann = {"keypoints": [10, 10, 2, 30, 30, 2]}
    assert ImageLabel._annotation_contains(ann, (20, 20)) is True
    assert ImageLabel._annotation_contains(ann, (40, 40)) is False


def test_translate_keypoints_keeps_visibility():
    out = ImageLabel._translate_keypoints([10, 20, 2, 30, 40, 1], 5, -5)
    assert out == [15, 15, 2, 35, 35, 1]


def test_scale_keypoints_doubles_within_box_keeps_visibility():
    # Scale box (0,0,10,10) -> (0,0,20,20): every coord doubles, v untouched.
    out = ImageLabel._scale_keypoints([0, 0, 2, 10, 10, 1], (0, 0, 10, 10), (0, 0, 20, 20))
    assert out == [0, 0, 2, 20, 20, 1]


def test_translate_keypoints_leaves_not_labelled_points_at_origin():
    # A v=0 padding point must stay at (0, 0) — COCO/YOLO-pose require it, and
    # transforming it would plant a bogus but plausible-looking coordinate.
    out = ImageLabel._translate_keypoints([10, 20, 2, 0, 0, 0], 5, -5)
    assert out == [15, 15, 2, 0, 0, 0]


def test_scale_keypoints_leaves_not_labelled_points_at_origin():
    out = ImageLabel._scale_keypoints([0, 0, 2, 0, 0, 0], (0, 0, 10, 10), (0, 0, 20, 20))
    assert out == [0, 0, 2, 0, 0, 0]
