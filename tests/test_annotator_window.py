"""
Tests for the ImageAnnotator class.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import unittest
from PyQt5.QtWidgets import QApplication
from image_annotator.annotator_window import ImageAnnotator

class TestImageAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.annotator = ImageAnnotator()

    def test_initialization(self):
        self.assertIsNotNone(self.annotator.image_label)
        self.assertIsNotNone(self.annotator.class_list)
        self.assertIsNotNone(self.annotator.annotation_list)

    # Add more test methods here

if __name__ == '__main__':
    unittest.main()
