"""
Tests for utility functions.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import unittest
from digitalsreeni_image_annotator.utils import calculate_area, calculate_bbox

class TestUtils(unittest.TestCase):
    def test_calculate_area(self):
        annotation = {"segmentation": [0, 0, 0, 1, 1, 1, 1, 0]}
        self.assertEqual(calculate_area(annotation), 1.0)

    def test_calculate_bbox(self):
        segmentation = [0, 0, 0, 1, 1, 1, 1, 0]
        self.assertEqual(calculate_bbox(segmentation), [0, 0, 1, 1])

    # Add more test methods here

if __name__ == '__main__':
    unittest.main()
