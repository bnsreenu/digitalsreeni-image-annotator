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
from PyQt6.QtGui import QImage
import yaml as yaml_lib
from src.digitalsreeni_image_annotator.io.export_formats import (
    export_coco_json,
    export_yolo_v4,
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
    image = QImage(100, 100, QImage.Format.Format_RGB32)
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
def sample_keypoint_schema():
    """A 3-point schema with a skeleton edge and a non-identity flip map."""
    return {"names": ["nose", "l_eye", "r_eye"], "skeleton": [[0, 1], [0, 2]], "flip_idx": [0, 2, 1]}


@pytest.fixture
def sample_pose_class_mapping():
    return {"person": 1}


@pytest.fixture
def sample_keypoint_annotations():
    """One 3-keypoint pose instance (K matches sample_keypoint_schema)."""
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

    def test_export_coco_category_includes_keypoints_skeleton_flip_idx(
        self, temp_output_dir, sample_class_mapping, sample_image_paths,
        sample_keypoint_schema,
    ):
        """A pose class's category gains keypoints/skeleton (1-based)/
        flip_idx (0-based); a plain class's category is unaffected (#35 PR-2)."""
        json_file_path, _ = export_coco_json(
            {},
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir,
            keypoint_schemas={"cell": sample_keypoint_schema},
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        categories = {c["name"]: c for c in coco_data["categories"]}
        cell_cat = categories["cell"]
        assert cell_cat["keypoints"] == ["nose", "l_eye", "r_eye"]
        assert cell_cat["skeleton"] == [[1, 2], [1, 3]]
        assert cell_cat["flip_idx"] == [0, 2, 1]

        nucleus_cat = categories["nucleus"]
        assert "keypoints" not in nucleus_cat
        assert "skeleton" not in nucleus_cat
        assert "flip_idx" not in nucleus_cat

    def test_export_coco_no_schemas_matches_legacy_output(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths,
    ):
        """Backward compat: no pose classes -> categories are byte-identical
        {id, name} dicts, same as before PR-2."""
        json_file_path, _ = export_coco_json(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir,
        )

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

        for category in coco_data["categories"]:
            assert set(category.keys()) == {"id", "name"}


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

    def test_create_coco_annotation_with_keypoints(self, sample_class_mapping):
        """A keypoint instance gets keypoints/num_keypoints/bbox, no
        segmentation, and an int visibility flag (issue #35 PR-2)."""
        ann = {
            "keypoints": [10.5, 20.5, 2.0, 30.5, 40.5, 0.0],
            "bbox": [5, 5, 30, 40],
        }

        coco_ann = create_coco_annotation(ann, image_id=1, annotation_id=1,
                                         class_name="cell", class_mapping=sample_class_mapping)

        assert coco_ann["keypoints"] == [10.5, 20.5, 2, 30.5, 40.5, 0]
        assert isinstance(coco_ann["keypoints"][2], int)
        assert isinstance(coco_ann["keypoints"][5], int)
        assert coco_ann["num_keypoints"] == 1  # only one point has v > 0
        assert coco_ann["bbox"] == [5, 5, 30, 40]
        assert "segmentation" not in coco_ann

    def test_create_coco_annotation_keypoints_checked_before_bbox(self, sample_class_mapping):
        """An ann carrying both keypoints and bbox takes the keypoints
        branch — regression guard for the elif ordering (#35 PR-2)."""
        ann = {
            "keypoints": [1, 2, 2],
            "num_keypoints": 1,
            "bbox": [0, 0, 5, 5],
        }

        coco_ann = create_coco_annotation(ann, image_id=1, annotation_id=1,
                                         class_name="cell", class_mapping=sample_class_mapping)

        assert "keypoints" in coco_ann


class TestYOLOExport:
    """Tests for YOLO export format."""

    def test_export_yolo_creates_directories(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that YOLO export creates the documented images/labels x
        train/val directory structure (see export_yolo_v5plus docstring)."""
        export_yolo_v5plus(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        for sub in [
            os.path.join('images', 'train'),
            os.path.join('images', 'val'),
            os.path.join('labels', 'train'),
            os.path.join('labels', 'val'),
        ]:
            full = os.path.join(temp_output_dir, sub)
            assert os.path.isdir(full), f"Expected directory not created: {sub}"

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

    def _multi_image_dataset(self, temp_output_dir, sample_image, count=5):
        """Build `count` annotated images with real files on disk.

        Returns (annotations, image_paths, src_dir) — src_dir is a separate
        directory so the export output never collides with the sources.
        """
        src_dir = os.path.join(temp_output_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        annotations = {}
        image_paths = {}
        for i in range(count):
            name = f"img_{i:03d}.png"
            path = os.path.join(src_dir, name)
            sample_image.save(path)
            image_paths[name] = path
            annotations[name] = {
                "cell": [
                    {"segmentation": [10, 10, 40, 10, 40, 40, 10, 40], "category": "cell"}
                ]
            }
        return annotations, image_paths

    def test_export_yolo_splits_train_and_val(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """A non-zero split populates both images/train and images/val, and
        each label .txt sits beside its image in the same split (issue #83)."""
        out_dir = os.path.join(temp_output_dir, "out")
        annotations, image_paths = self._multi_image_dataset(
            temp_output_dir, sample_image, count=5
        )

        export_yolo_v5plus(
            annotations, sample_class_mapping, image_paths,
            slices=[], image_slices={}, output_dir=out_dir, val_split=40,
        )

        train_imgs = os.listdir(os.path.join(out_dir, 'images', 'train'))
        val_imgs = os.listdir(os.path.join(out_dir, 'images', 'val'))
        assert len(train_imgs) > 0 and len(val_imgs) > 0
        assert len(train_imgs) + len(val_imgs) == 5
        assert len(val_imgs) == 2  # 40% of 5

        # Every image has its label beside it in the matching split.
        for split, imgs in (('train', train_imgs), ('val', val_imgs)):
            label_dir = os.path.join(out_dir, 'labels', split)
            for img in imgs:
                stem = os.path.splitext(img)[0]
                assert os.path.exists(os.path.join(label_dir, stem + '.txt'))

    def test_export_yolo_zero_split_leaves_val_empty(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """val_split=0 preserves the historical all-in-train behaviour."""
        out_dir = os.path.join(temp_output_dir, "out0")
        annotations, image_paths = self._multi_image_dataset(
            temp_output_dir, sample_image, count=4
        )

        export_yolo_v5plus(
            annotations, sample_class_mapping, image_paths,
            slices=[], image_slices={}, output_dir=out_dir, val_split=0,
        )

        train_imgs = os.listdir(os.path.join(out_dir, 'images', 'train'))
        val_imgs = os.listdir(os.path.join(out_dir, 'images', 'val'))
        assert len(train_imgs) == 4
        assert len(val_imgs) == 0

    def test_export_yolo_splits_multidim_slices(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """Multi-dim slices (no file path, carried as QImage in `slices`) must
        also route across the train/val split, not all land in train."""
        out_dir = os.path.join(temp_output_dir, "out_slices")
        slices = []
        annotations = {}
        for i in range(5):
            slice_name = f"stack.tif_T0_Z{i}_C0"
            slices.append((slice_name, sample_image))
            annotations[slice_name] = {
                "cell": [
                    {"segmentation": [10, 10, 40, 10, 40, 40, 10, 40], "category": "cell"}
                ]
            }

        export_yolo_v5plus(
            annotations, sample_class_mapping, image_paths={},
            slices=slices, image_slices={}, output_dir=out_dir, val_split=40,
        )

        train_imgs = os.listdir(os.path.join(out_dir, 'images', 'train'))
        val_imgs = os.listdir(os.path.join(out_dir, 'images', 'val'))
        assert len(train_imgs) > 0 and len(val_imgs) > 0
        assert len(train_imgs) + len(val_imgs) == 5
        assert len(val_imgs) == 2  # 40% of 5

        # Each slice's label sits in the same split as its image.
        for split, imgs in (('train', train_imgs), ('val', val_imgs)):
            label_dir = os.path.join(out_dir, 'labels', split)
            for img in imgs:
                stem = os.path.splitext(img)[0]
                assert os.path.exists(os.path.join(label_dir, stem + '.txt'))

    def test_export_yolo_single_image_val_falls_back_to_train(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """A single-image project can't fill val, so data.yaml's val must fall
        back to the (populated) train dir rather than an empty images/val."""
        out_dir = os.path.join(temp_output_dir, "out_single")
        annotations, image_paths = self._multi_image_dataset(
            temp_output_dir, sample_image, count=1
        )

        export_yolo_v5plus(
            annotations, sample_class_mapping, image_paths,
            slices=[], image_slices={}, output_dir=out_dir, val_split=20,
        )

        val_imgs = os.listdir(os.path.join(out_dir, 'images', 'val'))
        assert len(val_imgs) == 0  # nothing could be routed to val
        with open(os.path.join(out_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)
        assert yaml_data['val'] == os.path.join('images', 'train')
        # path + relative val must resolve to a real, populated directory.
        resolved_val = os.path.join(yaml_data['path'], yaml_data['val'])
        assert os.path.isdir(resolved_val)
        assert len(os.listdir(resolved_val)) > 0

    def test_export_yolo_v4_splits_train_and_valid(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """YOLO v4 routes images into train/ and valid/ and points data.yaml's
        val at the populated valid/ images dir when a split is requested."""
        out_dir = os.path.join(temp_output_dir, "out_v4")
        annotations, image_paths = self._multi_image_dataset(
            temp_output_dir, sample_image, count=5
        )

        export_yolo_v4(
            annotations, sample_class_mapping, image_paths,
            slices=[], image_slices={}, output_dir=out_dir, val_split=40,
        )

        train_imgs = os.listdir(os.path.join(out_dir, 'train', 'images'))
        valid_imgs = os.listdir(os.path.join(out_dir, 'valid', 'images'))
        assert len(train_imgs) == 3 and len(valid_imgs) == 2

        with open(os.path.join(out_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)
        # val must resolve to the populated valid/images dir.
        assert os.path.basename(os.path.dirname(yaml_data['val'])) == 'valid'
        assert len(os.listdir(yaml_data['val'])) == 2

    def test_export_yolo_v4_zero_split_val_points_at_train(
        self, temp_output_dir, sample_class_mapping, sample_image
    ):
        """With no split, v4 keeps the historical behaviour: val points at the
        train images so the path is non-empty."""
        out_dir = os.path.join(temp_output_dir, "out_v4_zero")
        annotations, image_paths = self._multi_image_dataset(
            temp_output_dir, sample_image, count=3
        )

        export_yolo_v4(
            annotations, sample_class_mapping, image_paths,
            slices=[], image_slices={}, output_dir=out_dir, val_split=0,
        )

        assert len(os.listdir(os.path.join(out_dir, 'valid', 'images'))) == 0
        with open(os.path.join(out_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)
        assert os.path.basename(os.path.dirname(yaml_data['val'])) == 'train'


class TestYOLOPoseExport:
    """Tests for YOLO-pose keypoint export (issue #35 PR-2)."""

    def _train_label_line(self, out_dir):
        labels_dir = os.path.join(out_dir, 'labels', 'train')
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        assert len(label_files) == 1
        with open(os.path.join(labels_dir, label_files[0])) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        return lines[0]

    def test_export_yolo_pose_line_token_count(
        self, temp_output_dir, sample_pose_class_mapping, sample_image_paths,
        sample_keypoint_annotations, sample_keypoint_schema,
    ):
        """A K=3 pose line has 5 + 3*3 = 14 tokens, correctly normalized."""
        export_yolo_v5plus(
            sample_keypoint_annotations, sample_pose_class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=temp_output_dir,
            keypoint_schemas={"person": sample_keypoint_schema},
        )

        tokens = self._train_label_line(temp_output_dir).split()
        assert len(tokens) == 5 + 3 * 3
        assert tokens[0] == "0"  # class index
        # bbox = [5, 5, 20, 20] on a 100x100 image -> center (15,15), size (20,20)
        assert float(tokens[1]) == pytest.approx(0.15)
        assert float(tokens[2]) == pytest.approx(0.15)
        assert float(tokens[3]) == pytest.approx(0.20)
        assert float(tokens[4]) == pytest.approx(0.20)
        # first keypoint: (10, 10, v=2)
        assert float(tokens[5]) == pytest.approx(0.10)
        assert float(tokens[6]) == pytest.approx(0.10)
        assert tokens[7] == "2"

    def test_export_yolo_pose_yaml_has_kpt_shape_and_flip_idx(
        self, temp_output_dir, sample_pose_class_mapping, sample_image_paths,
        sample_keypoint_annotations, sample_keypoint_schema,
    ):
        export_yolo_v5plus(
            sample_keypoint_annotations, sample_pose_class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=temp_output_dir,
            keypoint_schemas={"person": sample_keypoint_schema},
        )

        with open(os.path.join(temp_output_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)

        assert yaml_data['kpt_shape'] == [3, 3]
        assert yaml_data['flip_idx'] == [0, 2, 1]

    def test_export_yolo_ordinary_export_omits_kpt_shape(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths,
    ):
        """A dataset with no keypoints gets no kpt_shape/flip_idx keys at
        all (backward compat)."""
        export_yolo_v5plus(
            sample_annotations, sample_class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=temp_output_dir,
        )

        with open(os.path.join(temp_output_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)

        assert 'kpt_shape' not in yaml_data
        assert 'flip_idx' not in yaml_data

    def test_export_yolo_pose_works_without_keypoint_schemas_arg(
        self, temp_output_dir, sample_pose_class_mapping, sample_image_paths,
        sample_keypoint_annotations,
    ):
        """Data-driven detection: annotations carry keypoints even when
        keypoint_schemas is omitted -> kpt_shape still written, flip_idx
        defaults to identity."""
        export_yolo_v5plus(
            sample_keypoint_annotations, sample_pose_class_mapping, sample_image_paths,
            slices=[], image_slices={}, output_dir=temp_output_dir,
        )

        with open(os.path.join(temp_output_dir, 'data.yaml')) as f:
            yaml_data = yaml_lib.safe_load(f)

        assert yaml_data['kpt_shape'] == [3, 3]
        assert yaml_data['flip_idx'] == [0, 1, 2]

    def test_export_yolo_refuses_mixed_k_schemas(self, temp_output_dir, sample_image_paths):
        """Two pose classes with different K must be refused -- a dataset's
        data.yaml has one global kpt_shape, not one per class."""
        annotations = {
            "test_image.png": {
                "person": [{"keypoints": [1, 1, 2] * 2, "bbox": [0, 0, 5, 5], "category_id": 1}],
                "animal": [{"keypoints": [1, 1, 2] * 4, "bbox": [0, 0, 5, 5], "category_id": 2}],
            }
        }
        class_mapping = {"person": 1, "animal": 2}

        with pytest.raises(ValueError):
            export_yolo_v5plus(
                annotations, class_mapping, sample_image_paths,
                slices=[], image_slices={}, output_dir=temp_output_dir,
            )

        assert not os.path.exists(os.path.join(temp_output_dir, 'data.yaml'))
        assert not os.path.exists(os.path.join(temp_output_dir, 'images'))

    def test_export_yolo_refuses_pose_plus_nonpose_class(self, temp_output_dir, sample_image_paths):
        """A pose class mixed with a plain (non-keypoint) class in the same
        export must also be refused."""
        annotations = {
            "test_image.png": {
                "person": [{"keypoints": [1, 1, 2] * 3, "bbox": [0, 0, 5, 5], "category_id": 1}],
                "cell": [{"segmentation": [0, 0, 5, 0, 5, 5, 0, 5], "category_id": 2}],
            }
        }
        class_mapping = {"person": 1, "cell": 2}

        with pytest.raises(ValueError):
            export_yolo_v5plus(
                annotations, class_mapping, sample_image_paths,
                slices=[], image_slices={}, output_dir=temp_output_dir,
            )

        assert not os.path.exists(os.path.join(temp_output_dir, 'data.yaml'))


class TestPascalVOCExport:
    """Tests for Pascal VOC export format."""

    def test_export_pascal_voc_creates_directories(
        self, temp_output_dir, sample_annotations, sample_class_mapping,
        sample_image_paths
    ):
        """Test that Pascal VOC export creates the Annotations/ and images/
        directories at the output root (see export_pascal_voc_bbox)."""
        export_pascal_voc_bbox(
            sample_annotations,
            sample_class_mapping,
            sample_image_paths,
            slices=[],
            image_slices={},
            output_dir=temp_output_dir
        )

        annotations_dir = os.path.join(temp_output_dir, 'Annotations')
        images_dir = os.path.join(temp_output_dir, 'images')

        assert os.path.isdir(annotations_dir)
        assert os.path.isdir(images_dir)

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
