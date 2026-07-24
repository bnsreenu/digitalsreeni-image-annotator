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


@pytest.fixture
def make_test_video():
    """Factory: write a tiny MJPG/.avi video and return its path (issue #47).

    Shared by the video-handler unit tests and the video-loading integration
    tests. Frame ``i`` is filled BGR ``(0, 0, 10*i)`` — a pure red-channel
    ramp — so a forgotten ``cvtColor`` (BGR→RGB) regression is detectable:
    the decoded pixel would come back blue instead of red. MJPG on a uniform
    fill round-trips exactly, so frame ``i``'s red component reads back as
    ``10*i``.
    """
    import cv2
    import numpy as np

    def _make(dir_path, name="clip.avi", frames=8, width=32, height=24, fps=10.0):
        path = os.path.join(str(dir_path), name)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        assert writer.isOpened(), "cv2.VideoWriter failed to open (MJPG/.avi)"
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 2] = 10 * i  # BGR: ramp the red channel
            writer.write(frame)
        writer.release()
        return path

    return _make


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
