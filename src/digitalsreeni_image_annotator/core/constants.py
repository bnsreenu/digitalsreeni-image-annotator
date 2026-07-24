"""
Constants for the Image Annotator application.

This module contains constant values used across the application.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

# File dialog filters
# (image/video open filter lives in video_handler.file_dialog_filter(), derived
#  from VIDEO_EXTS so it can't drift from is_video() -- #47/#48)
JSON_FILE_FILTER = "JSON Files (*.json)"

# Default window size
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 800

# Zoom settings
MIN_ZOOM = 10
MAX_ZOOM = 500
DEFAULT_ZOOM = 100

# Annotation settings
# Mask fill alpha — kept low so the underlying image stays legible through
# overlapping masks (the border still carries the class colour).
DEFAULT_FILL_OPACITY = 0.2

# Annotations panel table columns (issue #24). Column 0 (ID) carries the
# annotation dict in its UserRole — the value-equality marker the canvas ↔ list
# selection bridge (ADR-022) reads.
ANNOT_COL_ID = 0
ANNOT_COL_CLASS = 1
ANNOT_COL_AREA = 2
ANNOT_COL_DETAIL = 3

# Default class colour palette (tab10-style, moderately muted so masks don't
# overpower the image). Red is intentionally LAST so a fresh project's first
# class isn't red — selection highlighting is class-colour-independent, but
# starting on red was needlessly harsh. Hex strings keep this module Qt-free.
DEFAULT_CLASS_COLORS = [
    "#1F77B4",  # blue
    "#FF7F0E",  # orange
    "#2CA02C",  # green
    "#9467BD",  # purple
    "#17BECF",  # cyan
    "#BCBD22",  # olive
    "#E377C2",  # pink
    "#8C564B",  # brown
    "#7F7F7F",  # gray
    "#D62728",  # red (last)
]


def default_class_color(index: int) -> str:
    """Hex colour for the index-th class, cycling through DEFAULT_CLASS_COLORS.

    Callers wrap the result in ``QColor(...)`` (kept out of this module so the
    core stays Qt-free)."""
    return DEFAULT_CLASS_COLORS[index % len(DEFAULT_CLASS_COLORS)]

