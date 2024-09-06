"""
Basic usage example for the Image Annotator package.
@DigitalSreeni
Dr. Sreenivas Bhattiprolu

Note: Once the package is installed, you can run the application directly from the command line by typing:
    digitalsreeni-image-annotator

This script demonstrates how to launch the application programmatically if needed.
"""

from digitalsreeni_image_annotator import ImageAnnotator
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()