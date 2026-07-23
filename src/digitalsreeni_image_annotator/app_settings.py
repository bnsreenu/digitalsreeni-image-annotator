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
