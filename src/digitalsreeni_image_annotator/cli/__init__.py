"""Headless command-line interface (issue #76, ADR-041).

Everything the app could do required a human, a screen and a mouse. There was
no way to regenerate a training dataset in a build script, convert between
annotation formats without opening the GUI, fail a CI job because someone
committed self-intersecting polygons, or run a model over a folder overnight.
For an ML engineer, an annotation tool that cannot be scripted is a tool that
has to be baby-sat.

**Nothing in this package may import Qt at module level, and a test enforces
it.** The import guard is the crux of the whole issue: an accidental
``from PyQt6 ...`` in a shared module would silently make headless validation
require a display, and it would work perfectly on the machine of whoever added
it. ``main.py`` also imports torch eagerly before ``QApplication`` to work
around a Windows DLL conflict (ADR-017); the CLI must not inherit that startup
path either.

The heavy stack is imported **lazily per command**: ``predict`` needs torch and
Ultralytics, ``export``/``convert``/``validate`` do not, and ``validate`` in
particular has to stay fast because it is the one that runs on every commit.
"""

from .main import main

__all__ = ["main"]
