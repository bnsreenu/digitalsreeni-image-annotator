"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import os
import sys
from PyQt5.QtWidgets import QApplication
from .annotator_window import ImageAnnotator

if sys.platform.startswith("linux"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def main():
    """
    Main function to run the Image Annotator application.
    """
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
