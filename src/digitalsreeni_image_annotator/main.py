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

# ── Qt import guard (issue #92, ADR-046)
#
# This block MUST stay below the torch import above.  ADR-017 requires torch to
# claim its DLL slot before any Qt DLL loads, and this is what loads Qt; the two
# Windows DLL workarounds now sit adjacent, and reordering them re-breaks ADR-017
# in a way that only shows up on Windows with torch installed.
#
# In a contaminated environment -- typically Conda, where a stray Qt6Core.dll or an
# outdated msvcp140.dll shadows the one the PyQt6 wheel ships -- this import fails
# with a bare "DLL load failed while importing QtCore: The specified procedure could
# not be found", which tells the user nothing about which DLL or what to do.
# core.qt_diagnostics names the offending file instead.  It cannot help when the
# loader terminates the process outright rather than raising; that is what
# `sreeni-cli doctor` is for, since the CLI never imports Qt at all.
#
# Only the PyQt6 import is guarded.  Wrapping the annotator_window import too would
# report an unrelated missing dependency as a Qt failure.
try:
    from PyQt6.QtWidgets import QApplication
except ImportError as exc:
    from .core.qt_diagnostics import format_import_failure
    print(format_import_failure(exc), file=sys.stderr)
    sys.exit(1)

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
