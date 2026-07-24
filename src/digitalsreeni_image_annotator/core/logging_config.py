"""Central logging configuration for the application.

See the "Logging and Debug Output" section in
``docs/08_crosscutting_concepts.md`` and ADR-030 for the policy. The short
version: the whole package logs through one stdlib ``logging`` tree rooted at
``digitalsreeni_image_annotator``, configured once with a single stderr
handler. Level defaults to INFO; ``--debug`` on the command line or
``IMAGE_ANNOTATOR_DEBUG=1`` in the environment switches it to DEBUG.

Application code must use ``get_logger(__name__)`` and never a bare ``print``.
"""
import logging
import os
import sys

# Derived from this module's own dotted name so the handler attaches to the SAME
# package root the app was imported under -- ``digitalsreeni_image_annotator``
# (installed / the ``sreeni`` entry point) OR ``src.digitalsreeni_image_annotator``
# (the documented ``python -m src.digitalsreeni_image_annotator.main`` launcher).
# This module lives at ``<root>.core.logging_config``, so the root is ``__name__``
# minus its last two components. Hardcoding one string silently dropped every
# record under the other import root.
_PKG = __name__.rsplit(".", 2)[0]


def configure(level=None):
    """Configure the package logger once. Idempotent — safe to call twice
    (tests, re-entry): a second call must not add a second handler."""
    root = logging.getLogger(_PKG)
    if root.handlers:          # already configured
        return root
    if level is None:
        debug = ("--debug" in sys.argv
                 or os.environ.get("IMAGE_ANNOTATOR_DEBUG", "") not in ("", "0"))
        level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()   # stderr
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return root


def get_logger(name):
    """Return a logger for ``name`` (pass ``__name__``). Every module shares the
    package root that :func:`configure` derives from ``__name__``, so the
    returned logger inherits that root's handler/level automatically — whether
    the app is imported as ``digitalsreeni_image_annotator`` or
    ``src.digitalsreeni_image_annotator``."""
    return logging.getLogger(name)
