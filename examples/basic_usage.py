"""
Basic usage example for the Image Annotator package.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

from image_annotator import ImageAnnotator
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()