"""
Unit tests for io_controller._rebuild_imported_annotation (issue #35 PR-2).

Regression coverage for the critical constraint: a keypoint-shaped result
must NEVER carry a `segmentation`/`type` key (even as None), because several
existence-only checks elsewhere (`"segmentation" in annotation`, not a
None-guard) would misfire on it -- see image_label.py::draw_annotations /
start_polygon_edit and eraser_tool.py. See ADR-029.
"""

import pytest

from src.digitalsreeni_image_annotator.controllers.io_controller import (
    _rebuild_imported_annotation,
)


def test_keypoint_annotation_has_no_segmentation_or_type_keys():
    ann = {
        "keypoints": [10, 10, 2, 20, 20, 1],
        "num_keypoints": 2,
        "bbox": [5, 5, 20, 20],
        "category_id": 1,
    }

    result = _rebuild_imported_annotation(ann, "person", 1)

    assert "segmentation" not in result
    assert "type" not in result
    assert result["keypoints"] == [10, 10, 2, 20, 20, 1]
    assert result["num_keypoints"] == 2
    assert result["bbox"] == [5, 5, 20, 20]
    assert result["category_name"] == "person"
    assert result["number"] == 1


def test_keypoint_annotation_computes_num_keypoints_when_missing():
    ann = {"keypoints": [10, 10, 2, 20, 20, 0], "bbox": [5, 5, 20, 20], "category_id": 1}

    result = _rebuild_imported_annotation(ann, "person", 1)

    assert result["num_keypoints"] == 1


def test_plain_annotation_matches_legacy_shape():
    ann = {
        "segmentation": [0, 0, 10, 0, 10, 10, 0, 10],
        "bbox": [0, 0, 10, 10],
        "category_id": 1,
        "type": "polygon",
    }

    result = _rebuild_imported_annotation(ann, "cell", 3)

    assert result == {
        "segmentation": [0, 0, 10, 0, 10, 10, 0, 10],
        "bbox": [0, 0, 10, 10],
        "category_id": 1,
        "category_name": "cell",
        "number": 3,
        "type": "polygon",
    }


def test_plain_annotation_not_marked_as_keypoint():
    """The regression the critical constraint requires: a plain import must
    never satisfy `"keypoints" in result` (even with a None value)."""
    ann = {"bbox": [0, 0, 10, 10], "category_id": 1, "type": "rectangle"}

    result = _rebuild_imported_annotation(ann, "cell", 1)

    assert "keypoints" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
