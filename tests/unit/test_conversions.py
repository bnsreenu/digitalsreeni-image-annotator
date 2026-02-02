"""
Unit tests for coordinate conversion functions.

Tests for screen-to-image and image-to-screen coordinate conversions.
"""

import pytest
import sys
import os
import importlib.util
from PyQt5.QtCore import QPoint, QSize
from PyQt5.QtGui import QPixmap

# Import image_label module directly by file path to avoid torch dependency issues
image_label_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'digitalsreeni_image_annotator', 'image_label.py')
spec = importlib.util.spec_from_file_location("image_label", image_label_path)
image_label = importlib.util.module_from_spec(spec)
sys.modules['digitalsreeni_image_annotator.image_label'] = image_label
spec.loader.exec_module(image_label)

ImageLabel = image_label.ImageLabel


@pytest.fixture
def image_label(qtbot):
    """Create an ImageLabel instance for testing."""
    label = ImageLabel(None)
    qtbot.addWidget(label)
    return label


@pytest.fixture
def image_label_with_image(qtbot):
    """Create an ImageLabel with a test image loaded."""
    label = ImageLabel(None)
    qtbot.addWidget(label)

    # Create a test pixmap (100x100)
    pixmap = QPixmap(100, 100)
    label.original_pixmap = pixmap
    label.scaled_pixmap = pixmap
    label.zoom_factor = 1.0
    label.offset_x = 0
    label.offset_y = 0

    return label


class TestGetImageCoordinates:
    """Tests for get_image_coordinates method."""

    def test_no_pixmap_returns_zero(self, image_label):
        """Test that get_image_coordinates returns (0,0) when no image is loaded."""
        pos = QPoint(50, 50)
        coords = image_label.get_image_coordinates(pos)
        assert coords == (0, 0)

    def test_identity_conversion_no_zoom_no_offset(self, image_label_with_image):
        """Test coordinate conversion with no zoom and no offset."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(50, 50)
        coords = label.get_image_coordinates(pos)
        assert coords == (50, 50)

    def test_conversion_with_offset(self, image_label_with_image):
        """Test coordinate conversion with offset (pan)."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 20
        label.offset_y = 30

        # Screen position (70, 80) -> Image position (50, 50)
        pos = QPoint(70, 80)
        coords = label.get_image_coordinates(pos)
        assert coords == (50, 50)

    def test_conversion_with_zoom_2x(self, image_label_with_image):
        """Test coordinate conversion with 2x zoom."""
        label = image_label_with_image
        label.zoom_factor = 2.0
        label.offset_x = 0
        label.offset_y = 0

        # Screen position (100, 100) -> Image position (50, 50)
        pos = QPoint(100, 100)
        coords = label.get_image_coordinates(pos)
        assert coords == (50, 50)

    def test_conversion_with_zoom_half(self, image_label_with_image):
        """Test coordinate conversion with 0.5x zoom (zoomed out)."""
        label = image_label_with_image
        label.zoom_factor = 0.5
        label.offset_x = 0
        label.offset_y = 0

        # Screen position (50, 50) -> Image position (100, 100)
        pos = QPoint(50, 50)
        coords = label.get_image_coordinates(pos)
        assert coords == (100, 100)

    def test_conversion_with_zoom_and_offset(self, image_label_with_image):
        """Test coordinate conversion with both zoom and offset."""
        label = image_label_with_image
        label.zoom_factor = 2.0
        label.offset_x = 50
        label.offset_y = 100

        # Screen position (150, 200) -> Image position (50, 50)
        # (150 - 50) / 2.0 = 50
        # (200 - 100) / 2.0 = 50
        pos = QPoint(150, 200)
        coords = label.get_image_coordinates(pos)
        assert coords == (50, 50)

    def test_conversion_origin(self, image_label_with_image):
        """Test coordinate conversion at origin."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(0, 0)
        coords = label.get_image_coordinates(pos)
        assert coords == (0, 0)

    def test_conversion_negative_after_offset(self, image_label_with_image):
        """Test coordinate conversion that results in negative image coordinates."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 100
        label.offset_y = 100

        # Screen position (50, 50) -> Image position (-50, -50)
        pos = QPoint(50, 50)
        coords = label.get_image_coordinates(pos)
        assert coords == (-50, -50)

    def test_conversion_fractional_coordinates(self, image_label_with_image):
        """Test coordinate conversion with fractional results (should round to int)."""
        label = image_label_with_image
        label.zoom_factor = 3.0
        label.offset_x = 0
        label.offset_y = 0

        # Screen position (10, 10) -> Image position (3.33, 3.33) -> (3, 3)
        pos = QPoint(10, 10)
        coords = label.get_image_coordinates(pos)
        assert coords == (3, 3)

    def test_conversion_large_zoom(self, image_label_with_image):
        """Test coordinate conversion with large zoom factor."""
        label = image_label_with_image
        label.zoom_factor = 10.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(500, 500)
        coords = label.get_image_coordinates(pos)
        assert coords == (50, 50)

    def test_conversion_small_zoom(self, image_label_with_image):
        """Test coordinate conversion with small zoom factor."""
        label = image_label_with_image
        label.zoom_factor = 0.1
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(10, 10)
        coords = label.get_image_coordinates(pos)
        assert coords == (100, 100)


class TestCoordinateConversionProperties:
    """Property-based tests for coordinate conversions."""

    def test_inverse_conversion(self, image_label_with_image):
        """Test that converting back and forth preserves coordinates (approximately)."""
        label = image_label_with_image
        label.zoom_factor = 2.0
        label.offset_x = 50
        label.offset_y = 100

        # Original screen position
        original_screen = QPoint(200, 300)

        # Convert to image coordinates
        image_coords = label.get_image_coordinates(original_screen)

        # Convert back to screen coordinates manually
        # screen_x = image_x * zoom_factor + offset_x
        # screen_y = image_y * zoom_factor + offset_y
        screen_x = image_coords[0] * label.zoom_factor + label.offset_x
        screen_y = image_coords[1] * label.zoom_factor + label.offset_y

        # Due to integer rounding, we allow small difference
        assert abs(screen_x - original_screen.x()) <= 1
        assert abs(screen_y - original_screen.y()) <= 1

    @pytest.mark.parametrize("zoom_factor,offset_x,offset_y", [
        (1.0, 0, 0),
        (2.0, 50, 50),
        (0.5, 100, 100),
        (4.0, -50, -50),
        (0.25, 200, 300),
    ])
    def test_origin_conversion(self, image_label_with_image, zoom_factor, offset_x, offset_y):
        """Test that origin conversion works correctly with various zoom and offset values."""
        label = image_label_with_image
        label.zoom_factor = zoom_factor
        label.offset_x = offset_x
        label.offset_y = offset_y

        # Screen position at offset should map to image origin
        pos = QPoint(int(offset_x), int(offset_y))
        coords = label.get_image_coordinates(pos)
        assert coords == (0, 0)


class TestEdgeCases:
    """Edge case tests for coordinate conversions."""

    def test_zero_zoom_factor(self, image_label_with_image):
        """Test behavior with zero zoom factor (edge case)."""
        label = image_label_with_image
        label.zoom_factor = 0.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(50, 50)
        # This will cause division by zero - should handle gracefully
        # In production, zoom_factor should never be 0, but test anyway
        with pytest.raises(ZeroDivisionError):
            label.get_image_coordinates(pos)

    def test_very_large_screen_coordinates(self, image_label_with_image):
        """Test with very large screen coordinates."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(10000, 10000)
        coords = label.get_image_coordinates(pos)
        assert coords == (10000, 10000)

    def test_negative_screen_coordinates(self, image_label_with_image):
        """Test with negative screen coordinates."""
        label = image_label_with_image
        label.zoom_factor = 1.0
        label.offset_x = 0
        label.offset_y = 0

        pos = QPoint(-50, -50)
        coords = label.get_image_coordinates(pos)
        assert coords == (-50, -50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
