"""App-global UI preferences persisted via QSettings.

First (and so far only) QSettings usage in the app — see ADR in
docs/09_architecture_decisions.md. UI preferences (font size, dark
mode) are per-user, not per-project, so they live here rather than in
the .iap project file. On Windows this writes to the registry under
HKCU\\Software\\DigitalSreeni\\ImageAnnotator.

All functions accept an optional QSettings instance so tests can pass
an INI-backed temp file instead of touching the real registry.
"""

from PyQt6.QtCore import QSettings

FONT_PT_MIN = 8
FONT_PT_MAX = 24
FONT_PT_DEFAULT = 10

_KEY_FONT_PT = "ui/font_pt"
_KEY_DARK_MODE = "ui/dark_mode"

# MLflow experiment tracking (issue #74). Tracking is always on; only the
# *destination* is configurable. An empty URI means "let the tracker resolve a
# default under <project>/mlruns".
_KEY_MLFLOW_URI = "tracking/mlflow_uri"
_KEY_MLFLOW_EXPERIMENT = "tracking/experiment_name"
MLFLOW_EXPERIMENT_DEFAULT = "image-annotator-training"

# Onion-skinning (issue #67). A viewing preference, not project data, so it
# belongs here next to the other UI prefs rather than in the .iap file.
_KEY_ONION_ENABLED = "ui/onion_enabled"
_KEY_ONION_OPACITY = "ui/onion_opacity"
_KEY_ONION_OFFSET = "ui/onion_offset"
_KEY_ONION_MODE = "ui/onion_mode"
_KEY_ONION_CONTENT = "ui/onion_content"


def clamp_font_pt(pt) -> int:
    """Coerce any stored/passed value to a usable point size.

    QSettings round-trips values as strings on some backends, and a
    hand-edited registry/INI can contain garbage — fall back to the
    default rather than crash at startup.
    """
    try:
        pt = int(pt)
    except (TypeError, ValueError):
        return FONT_PT_DEFAULT
    return max(FONT_PT_MIN, min(FONT_PT_MAX, pt))


def _settings() -> QSettings:
    return QSettings("DigitalSreeni", "ImageAnnotator")


def load_ui_prefs(settings=None) -> tuple[int, bool]:
    """Return (font_pt, dark_mode), with defaults (10, True)."""
    if settings is None:
        settings = _settings()
    font_pt = clamp_font_pt(settings.value(_KEY_FONT_PT, FONT_PT_DEFAULT))
    dark_mode = settings.value(_KEY_DARK_MODE, True, type=bool)
    return font_pt, dark_mode


def save_ui_prefs(font_pt, dark_mode, settings=None) -> None:
    if settings is None:
        settings = _settings()
    settings.setValue(_KEY_FONT_PT, clamp_font_pt(font_pt))
    settings.setValue(_KEY_DARK_MODE, bool(dark_mode))


def load_onion_prefs(settings=None) -> tuple[bool, float, int, str, str]:
    """Return ``(enabled, opacity, offset, mode, content)`` for onion-skinning.

    Every value is passed through the ``core.onion`` clamps, so a hand-edited
    registry or INI cannot produce an opacity of 0 (invisible ghost, decode
    cost still paid) or an offset that reaches past the stack.
    """
    from .core import onion

    if settings is None:
        settings = _settings()
    enabled = settings.value(_KEY_ONION_ENABLED, False, type=bool)
    opacity = onion.clamp_opacity(
        settings.value(_KEY_ONION_OPACITY, onion.DEFAULT_OPACITY)
    )
    offset = onion.clamp_offset(settings.value(_KEY_ONION_OFFSET, onion.DEFAULT_OFFSET))
    mode = onion.normalise_mode(
        settings.value(_KEY_ONION_MODE, onion.DEFAULT_MODE, type=str)
    )
    content = onion.normalise_content(
        settings.value(_KEY_ONION_CONTENT, onion.DEFAULT_CONTENT, type=str)
    )
    return bool(enabled), opacity, offset, mode, content


def save_onion_prefs(enabled, opacity, offset, mode, content, settings=None) -> None:
    from .core import onion

    if settings is None:
        settings = _settings()
    settings.setValue(_KEY_ONION_ENABLED, bool(enabled))
    settings.setValue(_KEY_ONION_OPACITY, onion.clamp_opacity(opacity))
    settings.setValue(_KEY_ONION_OFFSET, onion.clamp_offset(offset))
    settings.setValue(_KEY_ONION_MODE, onion.normalise_mode(mode))
    settings.setValue(_KEY_ONION_CONTENT, onion.normalise_content(content))


def load_mlflow_prefs(settings=None) -> tuple[str, str]:
    """Return (tracking_uri, experiment_name).

    Tracking itself is not optional — only its destination is. Defaults:
    empty URI (the tracker resolves <project>/mlruns) and the shared default
    experiment name.
    """
    if settings is None:
        settings = _settings()
    uri = settings.value(_KEY_MLFLOW_URI, "", type=str) or ""
    experiment = (
        settings.value(_KEY_MLFLOW_EXPERIMENT, MLFLOW_EXPERIMENT_DEFAULT, type=str)
        or MLFLOW_EXPERIMENT_DEFAULT
    )
    return uri, experiment


def save_mlflow_prefs(uri, experiment, settings=None) -> None:
    if settings is None:
        settings = _settings()
    settings.setValue(_KEY_MLFLOW_URI, (uri or "").strip())
    settings.setValue(
        _KEY_MLFLOW_EXPERIMENT, (experiment or "").strip() or MLFLOW_EXPERIMENT_DEFAULT
    )
