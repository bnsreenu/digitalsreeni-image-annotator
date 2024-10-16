"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from .annotator_window import ImageAnnotator

# To address Linux errors, by removing the QT_QPA_PLATFORM_PLUGIN_PATH 
# environment variable on Linux systems, which allows the application 
# to use the system's Qt platform plugins instead of potentially conflicting ones
if sys.platform.startswith("linux"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

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