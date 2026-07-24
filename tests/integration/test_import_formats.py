"""
Integration tests for import formats (COCO JSON, YOLO v4/v5+).

Keypoint / pose support added for issue #35 PR-2 — see test_export_formats.py
for the export-side counterparts and test_keypoint_export_import_roundtrip.py
for full round-trips.
"""

import json

import pytest
import yaml as yaml_lib
from PIL import Image

from src.digitalsreeni_image_annotator.io.import_formats import (
    import_coco_json,
    import_yolo_v4,
    import_yolo_v5plus,
)


def _write_png(path, width=100, height=100):
    Image.new("RGB", (width, height)).save(path)


class TestCOCOImportKeypoints:
    """Recovering per-class keypoint schemas and pose instances from a COCO
    JSON file's categories/annotations (issue #35 PR-2)."""

    def _coco_with_pose_category(self, tmp_path, keypoints_ann=None, extra_category_fields=None):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_png(images_dir / "img.png")

        category = {"id": 1, "name": "person", **(extra_category_fields or {})}
        ann = keypoints_ann or {
            "id": 1, "image_id": 1, "category_id": 1,
            "keypoints": [10, 10, 2, 20, 20, 1, 0, 0, 0],
            "num_keypoints": 2,
            "bbox": [5, 5, 20, 20],
        }
        coco = {
            "images": [{"id": 1, "file_name": "img.png", "width": 100, "height": 100}],
            "categories": [category],
            "annotations": [ann],
        }
        json_path = tmp_path / "annotations.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        return json_path

    def test_recovers_keypoint_schema(self, tmp_path):
        json_path = self._coco_with_pose_category(
            tmp_path,
            extra_category_fields={
                "keypoints": ["nose", "l_eye", "r_eye"],
                "skeleton": [[1, 2], [1, 3]],  # 1-based per COCO spec
                "flip_idx": [0, 2, 1],
            },
        )

        _, _, schemas = import_coco_json(str(json_path), {})

        assert schemas["person"] == {
            "names": ["nose", "l_eye", "r_eye"],
            "skeleton": [[0, 1], [0, 2]],  # converted back to 0-based
            "flip_idx": [0, 2, 1],
        }

    def test_drops_malformed_keypoint_schema(self, tmp_path):
        """Duplicate names sanitize_schema()'s to None -> skipped, no crash."""
        json_path = self._coco_with_pose_category(
            tmp_path,
            extra_category_fields={"keypoints": ["a", "a"], "skeleton": [], "flip_idx": [0, 1]},
        )

        imported, _, schemas = import_coco_json(str(json_path), {})

        assert "person" not in schemas
        # the annotation itself still imports fine
        assert "person" in imported["img.png"]

    def test_keypoint_annotation_skips_bbox_synthesis(self, tmp_path):
        json_path = self._coco_with_pose_category(tmp_path)

        imported, _, _ = import_coco_json(str(json_path), {})

        ann = imported["img.png"]["person"][0]
        assert ann["keypoints"] == [10.0, 10.0, 2.0, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0]
        assert ann["num_keypoints"] == 2
        assert ann["bbox"] == [5.0, 5.0, 20.0, 20.0]
        assert "segmentation" not in ann
        assert "type" not in ann

    def test_keypoint_annotation_with_mask_drops_mask_and_notes_it(self, tmp_path):
        """A real-world person_keypoints-style annotation can carry BOTH
        keypoints and a segmentation mask; the app's pose model has no mask
        (ADR-029) so it's dropped -- but that's a data reduction, not a
        silent one (issue #35 PR-2 review).

        The drop is now announced via ``logger.info`` (issue #33), not a
        ``print``; capture it with a handler attached directly to the module's
        own logger (resolved from ``import_coco_json.__module__`` so it matches
        this file's ``src.`` import path, not a differently-rooted logger)."""
        import logging
        json_path = self._coco_with_pose_category(
            tmp_path,
            keypoints_ann={
                "id": 1, "image_id": 1, "category_id": 1,
                "keypoints": [10, 10, 2, 20, 20, 1, 0, 0, 0],
                "num_keypoints": 2,
                "bbox": [5, 5, 20, 20],
                "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]],
            },
        )

        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())
        mod_logger = logging.getLogger(import_coco_json.__module__)
        old_level = mod_logger.level
        mod_logger.addHandler(handler)
        mod_logger.setLevel(logging.INFO)
        try:
            imported, _, _ = import_coco_json(str(json_path), {})
        finally:
            mod_logger.removeHandler(handler)
            mod_logger.setLevel(old_level)

        ann = imported["img.png"]["person"][0]
        assert "keypoints" in ann
        assert "segmentation" not in ann
        assert any("dropped" in m for m in records)

    def test_keypoint_annotation_missing_bbox_gets_synthesized(self, tmp_path):
        json_path = self._coco_with_pose_category(
            tmp_path,
            keypoints_ann={
                "id": 1, "image_id": 1, "category_id": 1,
                "keypoints": [10, 10, 2, 20, 20, 1, 0, 0, 0],
                "num_keypoints": 2,
                # no "bbox" key
            },
        )

        imported, _, _ = import_coco_json(str(json_path), {})

        ann = imported["img.png"]["person"][0]
        assert "bbox" in ann
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0
        # derived box must contain both labelled points (10,10) and (20,20)
        assert x <= 10 and y <= 10
        assert x + w >= 20 and y + h >= 20

    def test_plain_annotation_still_imports_normally(self, tmp_path):
        """A COCO file with no keypoints anywhere imports exactly as before,
        and the recovered schema dict is empty."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        _write_png(images_dir / "img.png")
        coco = {
            "images": [{"id": 1, "file_name": "img.png", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "cell"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}],
        }
        json_path = tmp_path / "annotations.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)

        imported, _, schemas = import_coco_json(str(json_path), {})

        assert schemas == {}
        ann = imported["img.png"]["cell"][0]
        assert "keypoints" not in ann
        # No source segmentation -> bbox->polygon synthesis, same as before PR-2.
        assert ann["type"] == "polygon"
        assert "segmentation" in ann


class TestYOLOv4ImportReturnContract:
    def test_returns_empty_schema_dict(self, tmp_path):
        """Legacy format stays detection-only; the 3-tuple contract must
        still hold (issue #35 PR-2)."""
        train_images = tmp_path / "train" / "images"
        train_labels = tmp_path / "train" / "labels"
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        _write_png(train_images / "img.png")
        (train_labels / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml_lib.safe_dump({"names": ["cell"]}), encoding="utf-8")

        imported, image_info, schemas = import_yolo_v4(str(yaml_path), {})

        assert schemas == {}
        assert "cell" in imported["img.png"]


class TestYOLOPoseImport:
    """Recovering kpt_shape/flip_idx and pose-shaped label lines from a
    YOLO-pose dataset (issue #35 PR-2)."""

    def _pose_dataset(self, tmp_path, class_names=("person",), kpt_shape=(3, 3), flip_idx=(0, 2, 1)):
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        _write_png(images_dir / "img.png", 100, 100)

        # class 0 box centered at (0.15, 0.15) size (0.2, 0.2), 3 keypoints
        line = "0 0.15 0.15 0.20 0.20  0.10 0.10 2  0.20 0.20 1  0.00 0.00 0"
        (labels_dir / "img.txt").write_text(line.replace("  ", " ") + "\n", encoding="utf-8")

        yaml_path = tmp_path / "data.yaml"
        yaml_data = {"names": list(class_names), "kpt_shape": list(kpt_shape), "flip_idx": list(flip_idx)}
        yaml_path.write_text(yaml_lib.safe_dump(yaml_data), encoding="utf-8")
        return yaml_path

    def test_recovers_kpt_shape_and_flip_idx_for_every_class(self, tmp_path):
        yaml_path = self._pose_dataset(tmp_path, class_names=("person", "animal"))

        _, _, schemas = import_yolo_v5plus(str(yaml_path), {})

        assert set(schemas.keys()) == {"person", "animal"}
        for schema in schemas.values():
            assert schema["names"] == ["kp0", "kp1", "kp2"]
            assert schema["skeleton"] == []
            assert schema["flip_idx"] == [0, 2, 1]
        # each class's schema is an independent dict, not aliased
        assert schemas["person"] is not schemas["animal"]

    def test_parses_pose_line_into_keypoints_and_bbox(self, tmp_path):
        yaml_path = self._pose_dataset(tmp_path)

        imported, _, _ = import_yolo_v5plus(str(yaml_path), {})

        ann = imported["img.png"]["person"][0]
        assert "keypoints" in ann
        assert ann["keypoints"] == pytest.approx([10.0, 10.0, 2.0, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0])
        assert ann["num_keypoints"] == 2
        x, y, w, h = ann["bbox"]
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(5.0)
        assert w == pytest.approx(20.0)
        assert h == pytest.approx(20.0)

    def test_ordinary_dataset_returns_empty_schema_dict(self, tmp_path):
        """No kpt_shape in the yaml -> ordinary bbox/polygon parsing,
        unchanged from before PR-2."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        _write_png(images_dir / "img.png")
        (labels_dir / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(yaml_lib.safe_dump({"names": ["cell"]}), encoding="utf-8")

        imported, _, schemas = import_yolo_v5plus(str(yaml_path), {})

        assert schemas == {}
        ann = imported["img.png"]["cell"][0]
        assert "keypoints" not in ann
        assert ann["type"] == "rectangle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
