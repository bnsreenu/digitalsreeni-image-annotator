"""
Tests for the ImageLabel class.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from digitalsreeni_image_annotator.image_label import ImageLabel

class TestImageLabel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.image_label = ImageLabel()

    def test_set_pixmap(self):
        pixmap = QPixmap(100, 100)
        self.image_label.setPixmap(pixmap)
        self.assertIsNotNone(self.image_label.original_pixmap)
        self.assertIsNotNone(self.image_label.scaled_pixmap)

    # Add more test methods here

if __name__ == '__main__':
    unittest.main()
