"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import sys
import os

# ── Windows DLL load-order workaround (torch → Qt, not Qt → torch)
#
# On Windows + Python 3.14, importing torch *after* PyQt has loaded
# its native platform DLLs (qwindows.dll via QtCore/Gui/Widgets)
# triggers WinError 1114 when torch's c10.dll initialises.  This
# was historically blamed on PyQt5 (ADR-011) and thought fixed in
# PyQt6 (ADR-014).  Real-world testing with torch 2.11.0 + PyQt6
# 6.10.2 shows the conflict still surfaces.  The workaround is
# cheap and harmless: import torch eagerly before QApplication is
# created so torch's DLLs claim their slot first.
# See ADR-017.
try:
    import torch  # noqa: F401
except ImportError:
    pass  # torch may not be installed; lazy fallback in sam_utils/dino_utils

from PyQt6.QtWidgets import QApplication
from .annotator_window import ImageAnnotator

# Legacy defensive cleanup from the PyQt5 era: a stale
# QT_QPA_PLATFORM_PLUGIN_PATH could shadow Qt's bundled XCB plugin and
# break startup on Linux. PyQt6 packaging is more robust about this, but
# the pop is cheap and harmless to keep.
if sys.platform.startswith("linux"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

def main():
    """
    Main function to run the Image Annotator application.
    """
    from .core.logging_config import configure
    configure()
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    # Offer to restore an unsaved-project recovery snapshot from a previous
    # session (issue #41). Done here — after show(), never in the constructor —
    # so tests that build ImageAnnotator() directly never trigger the modal.
    window.project_controller.offer_recovery()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
