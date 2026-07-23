"""
Pytest configuration file with common fixtures.
"""

import pytest
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure Qt platform for headless testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


@pytest.fixture(scope="session")
def qt_application():
    """Create a QApplication instance for the test session."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _drain_qt_deferred_deletes():
    """Actually destroy widgets that a test scheduled with ``deleteLater()``.

    The QApplication is session-scoped and never spins an event loop between
    tests, so ``deleteLater()`` alone never fires — every ImageAnnotator and
    dialog created by a fixture stays alive for the whole session. That
    accumulation eventually corrupts Qt state on Windows CI and segfaults with
    an access violation (seen in ``apply_theme_and_font`` while building the
    Nth window). Draining the posted ``DeferredDelete`` events after each test
    keeps at most one window alive at a time.

    Autouse + function-scoped, so this finalizes *after* the per-test ``window``
    fixture's ``deleteLater()`` runs — it drains exactly what that teardown just
    queued.
    """
    yield
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
