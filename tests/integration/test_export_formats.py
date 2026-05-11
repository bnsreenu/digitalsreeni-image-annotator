"""
Integration tests for export formats.

Tests for COCO JSON, YOLO, and Pascal VOC export functions.
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from PyQt5.QtGui import QImage
from src.digitalsreeni_image_annotator.export_formats import (
    export_coco_json,
    export_yolo_v5plus,
    export_pascal_voc_bbox,
    create_coco_annotation
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create a sample QImage for testing."""
    image = QImage(100, 100, QImage.Format_RGB32)
    image.fill(0xFFFFFFFF)  # White background
    return image


@pytest.fixture
def sample_annotations():
    """Create sample annotation data."""
    return {
        "test_image.png": {
            "cell": [
                {
                    "segmentation": [10, 10, 40, 10, 40, 40, 10, 40],
                    "category": "cell"
                },
                {
                    "segmentation": [60, 60, 80, 60, 80, 80, 60, 80],
                    "category": "cell"
                }
            ],
            "nucleus": [
                {
                    "bbox": [20, 20, 10, 10],  # x, y, width, height
                    "category": "nucleus"
                }
            ]
        }
    }


@pytest.fixture
def sample_class_mapping():
    """Create sample class mapping."""
    return {
        "cell": 1,
        "nucleus": 2
    }


@pytest.fixture
def sample_image_paths(temp_output_dir, sample_image):
    """Create sample image paths with actual test images."""
    image_path = os.path.join(temp_output_dir, "test_image.png")
    sample_image.save(image_path)
    return {
        "test_image.png": image_path
    }


class TestCOCOExport:
    """Tests for COCO JSON export format."""

    def test_export_coco_creates_output_directory(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export creates the output directory structure."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        assert os.path.exists(json_file_path)
        assert os.path.exists(images_dir)
        assert os.path.isdir(images_dir)

    def test_export_coco_creates_valid_json(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export creates valid JSON structure."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Check required keys
        assert "images" in coco_data
        assert "categories" in coco_data
        assert "annotations" in coco_data

        # Check data types
        assert isinstance(coco_data["images"], list)
        assert isinstance(coco_data["categories"], list)
        assert isinstance(coco_data["annotations"], list)

    def test_export_coco_categories(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export correctly exports categories."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        categories = coco_data["categories"]
        assert len(categories) == 2

        # Check category structure
        for category in categories:
            assert "id" in category
            assert "name" in category
            assert category["name"] in sample_class_mapping
            assert category["id"] == sample_class_mapping[category["name"]]

    def test_export_coco_images(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export correctly exports image information."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        images = coco_data["images"]
        assert len(images) == 1

        image_info = images[0]
        assert "id" in image_info
        assert "file_name" in image_info
        assert "height" in image_info
        assert "width" in image_info
        assert image_info["file_name"] == "test_image.png"
        assert image_info["height"] == 100
        assert image_info["width"] == 100

    def test_export_coco_annotations(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export correctly exports annotations."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        annotations = coco_data["annotations"]
        # 2 cell annotations + 1 nucleus annotation = 3 total
        assert len(annotations) == 3

        # Check annotation structure
        for ann in annotations:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "area" in ann
            assert "iscrowd" in ann
            assert ann["iscrowd"] == 0
            # Each annotation should have either segmentation or bbox
            assert "segmentation" in ann or "bbox" in ann

    def test_export_coco_copies_images(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export copies image files to output directory."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        copied_image_path = os.path.join(images_dir, "test_image.png")
        assert os.path.exists(copied_image_path)

    def test_export_coco_custom_filename(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that COCO export accepts custom JSON filename."""
        json_file_path, images_dir = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir,
            json_filename="custom_annotations.json"
        )

        assert os.path.basename(json_file_path) == "custom_annotations.json"

    def test_export_coco_empty_annotations(
        self, temp_output_dir, sample_class_mapping, sample_image_paths
    ):
        """Test COCO export with no annotations."""
        empty_annotations = {}
        json_file_path, images_dir = export_coco_json(
            empty_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        assert len(coco_data["images"]) == 0
        assert len(coco_data["annotations"]) == 0
        assert len(coco_data["categories"]) == 2  # Categories still present


class TestCreateCOCOAnnotation:
    """Tests for create_coco_annotation helper function."""

    def test_create_coco_annotation_with_segmentation(self, sample_class_mapping):
        """Test creating COCO annotation from segmentation."""
        ann = {
            "segmentation": [10, 10, 40, 10, 40, 40, 10, 40],
            "category": "cell"
        }

        coco_ann = create_coco_annotation(ann, image_id=1, annotation_id=1,
                                         class_name="cell", class_mapping=sample_class_mapping)

        assert coco_ann["id"] == 1
        assert coco_ann["image_id"] == 1
        assert coco_ann["category_id"] == 1
        assert "segmentation" in coco_ann
        assert "bbox" in coco_ann
        assert "area" in coco_ann
        assert coco_ann["iscrowd"] == 0

    def test_create_coco_annotation_with_bbox(self, sample_class_mapping):
        """Test creating COCO annotation from bounding box."""
        ann = {
            "bbox": [10, 10, 30, 30],
            "category": "nucleus"
        }

        coco_ann = create_coco_annotation(ann, image_id=2, annotation_id=2,
                                         class_name="nucleus", class_mapping=sample_class_mapping)

        assert coco_ann["id"] == 2
        assert coco_ann["image_id"] == 2
        assert coco_ann["category_id"] == 2
        assert "bbox" in coco_ann
        assert coco_ann["bbox"] == [10, 10, 30, 30]
        assert coco_ann["area"] == 900  # 30 * 30


class TestYOLOExport:
    """Tests for YOLO export format."""

    def test_export_yolo_creates_directories(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that YOLO export creates proper directory structure."""
        export_yolo_v5plus(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        train_dir = os.path.join(temp_output_dir, 'train')
        valid_dir = os.path.join(temp_output_dir, 'valid')

        assert os.path.exists(train_dir)
        assert os.path.exists(valid_dir)
        assert os.path.isdir(train_dir)
        assert os.path.isdir(valid_dir)

    def test_export_yolo_creates_yaml(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that YOLO export creates data.yaml file."""
        export_yolo_v5plus(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        yaml_path = os.path.join(temp_output_dir, 'data.yaml')
        assert os.path.exists(yaml_path)

    def test_export_yolo_yaml_content(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that YOLO data.yaml has correct content."""
        import yaml

        export_yolo_v5plus(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        yaml_path = os.path.join(temp_output_dir, 'data.yaml')
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        assert 'train' in yaml_data
        assert 'val' in yaml_data
        assert 'nc' in yaml_data
        assert 'names' in yaml_data
        assert yaml_data['nc'] == len(sample_class_mapping)
        assert isinstance(yaml_data['names'], list)


class TestPascalVOCExport:
    """Tests for Pascal VOC export format."""

    def test_export_pascal_voc_creates_directories(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that Pascal VOC export creates proper directory structure."""
        export_pascal_voc_bbox(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        annotations_dir = os.path.join(temp_output_dir, 'Annotations')
        images_dir = os.path.join(temp_output_dir, 'JPEGImages')

        assert os.path.exists(annotations_dir)
        assert os.path.exists(images_dir)

    def test_export_pascal_voc_creates_xml(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that Pascal VOC export creates XML annotation files."""
        export_pascal_voc_bbox(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        annotations_dir = os.path.join(temp_output_dir, 'Annotations')
        xml_files = list(Path(annotations_dir).glob('*.xml'))

        assert len(xml_files) > 0


class TestExportWithSlices:
    """Tests for export functions with multi-dimensional slices."""

    def test_export_coco_with_slices(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """Test COCO export with multi-dimensional image slices."""
        slice_name = "stack_T0_Z0_C0"
        annotations = {
            slice_name: {
                "cell": [
                    {
                        "segmentation": [10, 10, 40, 10, 40, 40, 10, 40],
                        "category": "cell"
                    }
                ]
            }
        }
        slices = [(slice_name, sample_image)]

        json_file_path, images_dir = export_coco_json(
            annotations,
            sample_class_mapping,
            image_paths={},
            slices=slices,
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        assert len(coco_data["images"]) == 1
        assert coco_data["images"][0]["file_name"] == f"{slice_name}.png"

        # Check that slice image was saved
        slice_image_path = os.path.join(images_dir, f"{slice_name}.png")
        assert os.path.exists(slice_image_path)


class TestExportEdgeCases:
    """Edge case tests for export functions."""

    def test_export_coco_with_no_categories(self, temp_output_dir, sample_image_paths):
        """Test COCO export with empty class mapping."""
        annotations = {}
        class_mapping = {}

        json_file_path, images_dir = export_coco_json(
            annotations,
            class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        assert len(coco_data["categories"]) == 0

    def test_export_coco_skips_images_without_annotations(
        self, temp_output_dir, sample_class_mapping, sample_image_paths
    ):
        """Test that images without annotations are skipped."""
        annotations = {
            "test_image.png": {}  # Empty annotations dict
        }

        json_file_path, images_dir = export_coco_json(
            annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        # Should skip the image since it has no annotations
        assert len(coco_data["images"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
