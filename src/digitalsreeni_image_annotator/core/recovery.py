"""Silent recovery-file autosave for unsaved projects (issue #41).

Work done before a project has ever been saved to disk is otherwise lost on a
crash or an accidental quit: ``auto_save()`` has no ``.iap`` to write to. This
module writes a snapshot of the current project state to an app-owned location
and remembers its path in QSettings, so the app can offer to restore it on the
next launch.

The snapshot is exactly a ``build_project_data()`` dict serialized like a real
``.iap`` (see ``ProjectController``), so restoring it needs no format-specific
code. Kept free of any main-window / Qt-widget coupling so it stays
unit-testable; every function accepts an optional ``QSettings`` instance,
mirroring ``app_settings``.
"""

import json
import os

from PyQt6.QtCore import QSettings, QStandardPaths

from . import image_utils
from .logging_config import get_logger

logger = get_logger(__name__)

# One stable recovery filename per install — a session overwrites it rather
# than accumulating stale snapshots. The `.iap.recovery` suffix keeps it from
# being mistaken for (or double-clicked as) a real project file.
_RECOVERY_FILENAME = "unsaved.iap.recovery"
_KEY_PENDING_PATH = "recovery/pending_path"


def _settings(settings=None) -> QSettings:
    # Same org/app as app_settings._settings() so every preference and the
    # recovery pointer live in one store.
    if settings is not None:
        return settings
    return QSettings("DigitalSreeni", "ImageAnnotator")


def recovery_dir() -> str:
    """App-owned writable dir for recovery snapshots, created on demand."""
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    path = os.path.join(base, "recovery")
    os.makedirs(path, exist_ok=True)
    return path


def write_recovery(project_data, settings=None) -> str:
    """Atomically write ``project_data`` as a recovery snapshot; remember its path.

    Written to a temp file then ``os.replace``d into place, so a crash mid-write
    can never truncate a previously-good snapshot. Returns the snapshot path.
    """
    path = os.path.join(recovery_dir(), _RECOVERY_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(image_utils.convert_to_serializable(project_data), f, indent=2)
    os.replace(tmp, path)
    _settings(settings).setValue(_KEY_PENDING_PATH, path)
    return path


def pending_recovery(settings=None):
    """Return the pending snapshot path iff the key and the file both exist.

    Otherwise clear a stale key and return ``None``.
    """
    s = _settings(settings)
    path = s.value(_KEY_PENDING_PATH, "", type=str)
    if path and os.path.exists(path):
        return path
    if path:
        s.remove(_KEY_PENDING_PATH)
    return None


def clear_recovery(settings=None) -> None:
    """Delete the snapshot (best-effort) and forget its path. Idempotent."""
    s = _settings(settings)
    path = s.value(_KEY_PENDING_PATH, "", type=str)
    if path:
        try:
            os.remove(path)
        except OSError:
            pass  # already gone / never written — nothing to reclaim
    s.remove(_KEY_PENDING_PATH)
