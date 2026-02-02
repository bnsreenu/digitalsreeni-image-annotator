"""
Unit tests for utility functions.

Tests for calculate_area, calculate_bbox, and normalize_image functions.
"""

import pytest
import numpy as np
import sys
import os
import importlib.util

# Import utils module directly by file path to avoid torch dependency issues
utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'digitalsreeni_image_annotator', 'utils.py')
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

calculate_area = utils.calculate_area
calculate_bbox = utils.calculate_bbox
normalize_image = utils.normalize_image


class TestCalculateArea:
    """Tests for calculate_area function."""

    def test_polygon_area_square(self):
        """Test area calculation for a square polygon."""
        # Square with side length 10 (area = 100)
        annotation = {
            "segmentation": [0, 0, 10, 0, 10, 10, 0, 10]
        }
        area = calculate_area(annotation)
        assert area == 100.0

    def test_polygon_area_rectangle(self):
        """Test area calculation for a rectangle polygon."""
        # Rectangle 20x5 (area = 100)
        annotation = {
            "segmentation": [0, 0, 20, 0, 20, 5, 0, 5]
        }
        area = calculate_area(annotation)
        assert area == 100.0

    def test_polygon_area_triangle(self):
        """Test area calculation for a triangle."""
        # Right triangle with base=10, height=10 (area = 50)
        annotation = {
            "segmentation": [0, 0, 10, 0, 0, 10]
        }
        area = calculate_area(annotation)
        assert area == 50.0

    def test_polygon_area_complex(self):
        """Test area calculation for a more complex polygon."""
        # L-shaped polygon
        annotation = {
            "segmentation": [0, 0, 10, 0, 10, 5, 5, 5, 5, 10, 0, 10]
        }
        area = calculate_area(annotation)
        # L-shape: 10x5 + 5x5 = 50 + 25 = 75
        assert area == 75.0

    def test_bbox_area(self):
        """Test area calculation for bounding box."""
        annotation = {
            "bbox": [10, 20, 30, 40]  # x, y, width, height
        }
        area = calculate_area(annotation)
        assert area == 1200.0  # 30 * 40

    def test_bbox_area_zero_width(self):
        """Test bounding box with zero width."""
        annotation = {
            "bbox": [10, 20, 0, 40]
        }
        area = calculate_area(annotation)
        assert area == 0.0

    def test_bbox_area_zero_height(self):
        """Test bounding box with zero height."""
        annotation = {
            "bbox": [10, 20, 30, 0]
        }
        area = calculate_area(annotation)
        assert area == 0.0

    def test_empty_annotation(self):
        """Test area calculation for annotation without segmentation or bbox."""
        annotation = {}
        area = calculate_area(annotation)
        assert area == 0

    def test_single_point_polygon(self):
        """Test area calculation for degenerate polygon (single point)."""
        annotation = {
            "segmentation": [5, 5]
        }
        area = calculate_area(annotation)
        assert area == 0.0


class TestCalculateBbox:
    """Tests for calculate_bbox function."""

    def test_bbox_from_square(self):
        """Test bounding box calculation from square polygon."""
        segmentation = [0, 0, 10, 0, 10, 10, 0, 10]
        bbox = calculate_bbox(segmentation)
        assert bbox == [0, 0, 10, 10]

    def test_bbox_from_triangle(self):
        """Test bounding box calculation from triangle."""
        segmentation = [5, 5, 15, 5, 10, 15]
        bbox = calculate_bbox(segmentation)
        assert bbox == [5, 5, 10, 10]

    def test_bbox_from_offset_polygon(self):
        """Test bounding box with non-zero origin."""
        segmentation = [100, 200, 150, 200, 150, 250, 100, 250]
        bbox = calculate_bbox(segmentation)
        assert bbox == [100, 200, 50, 50]

    def test_bbox_from_irregular_polygon(self):
        """Test bounding box from irregular polygon."""
        segmentation = [10, 20, 50, 15, 60, 45, 30, 55, 5, 40]
        bbox = calculate_bbox(segmentation)
        assert bbox == [5, 15, 55, 40]  # min_x=5, min_y=15, w=60-5=55, h=55-15=40

    def test_bbox_single_point(self):
        """Test bounding box from single point."""
        segmentation = [10, 20]
        bbox = calculate_bbox(segmentation)
        assert bbox == [10, 20, 0, 0]

    def test_bbox_horizontal_line(self):
        """Test bounding box from horizontal line."""
        segmentation = [10, 20, 50, 20]
        bbox = calculate_bbox(segmentation)
        assert bbox == [10, 20, 40, 0]

    def test_bbox_vertical_line(self):
        """Test bounding box from vertical line."""
        segmentation = [10, 20, 10, 60]
        bbox = calculate_bbox(segmentation)
        assert bbox == [10, 20, 0, 40]

    def test_bbox_negative_coordinates(self):
        """Test bounding box with negative coordinates."""
        segmentation = [-10, -20, 10, -20, 10, 20, -10, 20]
        bbox = calculate_bbox(segmentation)
        assert bbox == [-10, -20, 20, 40]

    def test_bbox_floating_point(self):
        """Test bounding box with floating point coordinates."""
        segmentation = [1.5, 2.5, 11.5, 2.5, 11.5, 12.5, 1.5, 12.5]
        bbox = calculate_bbox(segmentation)
        assert bbox == [1.5, 2.5, 10.0, 10.0]


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_uint16_to_uint8(self):
        """Test normalization of 16-bit image to 8-bit."""
        # 16-bit image with range 0-65535
        image = np.array([[0, 32767, 65535]], dtype=np.uint16)
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0] == 0
        assert normalized[0, 1] == 127  # Approximately middle
        assert normalized[0, 2] == 255

    def test_normalize_float_to_uint8(self):
        """Test normalization of float image to 8-bit."""
        image = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0] == 0
        assert normalized[0, 1] == 127  # Approximately middle
        assert normalized[0, 2] == 255

    def test_normalize_already_uint8(self):
        """Test that uint8 images are returned unchanged."""
        image = np.array([[0, 127, 255]], dtype=np.uint8)
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        np.testing.assert_array_equal(normalized, image)

    def test_normalize_2d_image(self):
        """Test normalization of 2D image."""
        image = np.array([
            [0, 100, 200],
            [300, 400, 500]
        ], dtype=np.uint16)
        normalized = normalize_image(image)

        assert normalized.shape == (2, 3)
        assert normalized.dtype == np.uint8
        assert normalized[0, 0] == 0  # Min value
        assert normalized[1, 2] == 255  # Max value

    def test_normalize_3d_image(self):
        """Test normalization of 3D image (e.g., RGB)."""
        image = np.array([
            [[0, 100, 200], [300, 400, 500]],
            [[600, 700, 800], [900, 1000, 1100]]
        ], dtype=np.uint16)
        normalized = normalize_image(image)

        assert normalized.shape == (2, 2, 3)
        assert normalized.dtype == np.uint8

    def test_normalize_constant_image(self):
        """Test normalization of constant image (all same value)."""
        # This will cause division by zero - test actual behavior
        image = np.array([[100, 100, 100]], dtype=np.uint16)

        # The function will divide by zero, resulting in nan, which converts to 0
        with np.errstate(invalid='ignore', divide='ignore'):
            normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        # After normalization with zero range, result should be 0 (nan -> 0)
        # Note: This might be a bug in the original implementation

    def test_normalize_negative_to_positive_range(self):
        """Test normalization of image with negative values."""
        image = np.array([[-100, 0, 100]], dtype=np.float64)
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0] == 0  # Min value
        assert normalized[0, 1] == 127  # Approximately middle
        assert normalized[0, 2] == 255  # Max value

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves image shape."""
        shapes = [(10, 10), (5, 20, 3), (100, 100)]
        for shape in shapes:
            image = np.random.randint(0, 1000, size=shape, dtype=np.uint16)
            normalized = normalize_image(image)
            assert normalized.shape == shape

    def test_normalize_small_range(self):
        """Test normalization with small value range."""
        image = np.array([[100, 101, 102]], dtype=np.uint16)
        normalized = normalize_image(image)

        assert normalized.dtype == np.uint8
        assert normalized[0, 0] == 0
        assert normalized[0, 2] == 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
