"""
Export -> import round-trip tests for keypoint/pose data (issue #35 PR-2).

Complements test_export_formats.py and test_import_formats.py, which test
each direction in isolation.
"""

import os

import pytest
from PyQt6.QtGui import QImage

from src.digitalsreeni_image_annotator.io.export_formats import export_coco_json, export_yolo_v5plus
from src.digitalsreeni_image_annotator.io.import_formats import import_coco_json, import_yolo_v5plus


@pytest.fixture
def sample_image_paths(tmp_path):
    image = QImage(100, 100, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    path = tmp_path / "src" / "test_image.png"
    path.parent.mkdir()
    image.save(str(path))
    return {"test_image.png": str(path)}


@pytest.fixture
def schema():
    return {"names": ["nose", "l_eye", "r_eye"], "skeleton": [[0, 1], [0, 2]], "flip_idx": [0, 2, 1]}


@pytest.fixture
def pose_annotations():
    return {
        "test_image.png": {
            "person": [
                {
                    "keypoints": [10, 10, 2, 20, 20, 1, 0, 0, 0],
                    "num_keypoints": 2,
                    "bbox": [5, 5, 20, 20],
                    "category_id": 1,
                    "category_name": "person",
                }
            ]
        }
    }


class TestCOCOKeypointRoundtrip:
    def test_schema_and_instance_survive(self, tmp_path, sample_image_paths, pose_annotations, schema):
        out_dir = str(tmp_path / "out")
        class_mapping = {"person": 1}

        json_file_path, _ = export_coco_json(
            pose_annotations, class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=out_dir,
            keypoint_schemas={"person": schema},
        )

        imported, _, recovered_schemas = import_coco_json(json_file_path, class_mapping)

        assert recovered_schemas["person"] == schema

        ann = imported["test_image.png"]["person"][0]
        assert ann["keypoints"] == pytest.approx([10, 10, 2, 20, 20, 1, 0, 0, 0])
        assert ann["num_keypoints"] == 2
        assert ann["bbox"] == pytest.approx([5, 5, 20, 20])
        assert "segmentation" not in ann


class TestYOLOPoseKeypointRoundtrip:
    def test_k_and_flip_idx_survive(self, tmp_path, sample_image_paths, pose_annotations, schema):
        out_dir = str(tmp_path / "out")
        class_mapping = {"person": 1}

        export_yolo_v5plus(
            pose_annotations, class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=out_dir,
            keypoint_schemas={"person": schema},
        )

        yaml_path = os.path.join(out_dir, "data.yaml")
        imported, _, recovered_schemas = import_yolo_v5plus(yaml_path, class_mapping)

        # YOLO-pose carries no point names -> generic kp0..kp{K-1}, but K
        # and flip_idx must survive exactly.
        assert recovered_schemas["person"]["names"] == ["kp0", "kp1", "kp2"]
        assert recovered_schemas["person"]["flip_idx"] == schema["flip_idx"]

        ann = imported["test_image.png"]["person"][0]
        assert ann["keypoints"] == pytest.approx([10, 10, 2, 20, 20, 1, 0, 0, 0], abs=1e-3)
        assert ann["bbox"] == pytest.approx([5, 5, 20, 20], abs=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
