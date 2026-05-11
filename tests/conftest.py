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
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
