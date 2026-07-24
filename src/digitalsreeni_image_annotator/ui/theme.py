"""Theme + font size application, extracted from `ImageAnnotator`.

The functions here take the main window as their first argument; they
read state directly off it (`dark_mode`, `ui_font_pt`, etc.) and write
to its widgets. Kept as plain functions rather than a controller class
because they are stateless and the call sites are sparse.

`mw.ui_font_pt` (int, 8-24, default 10) is the single source of truth
for UI text size. The Settings → Font Size presets jump to fixed
values; Ctrl+Shift+= / Ctrl+Shift+- step it ±1pt. Every change goes
through `set_font_pt`, which clamps, re-applies the theme, persists
via QSettings and syncs the preset menu checkmarks.
"""

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget

from ..app_settings import FONT_PT_DEFAULT, clamp_font_pt, save_ui_prefs
from .default_stylesheet import default_stylesheet
from .soft_dark_stylesheet import soft_dark_stylesheet

# Legacy px values from the static stylesheets at the 10pt default.
# The overrides scale these by ui_font_pt / 10 and stay in px so the
# default renders pixel-identical to the pre-zoom stylesheets.
_HEADER_PX_AT_DEFAULT = 14      # QLabel.section-header font-size
_INDICATOR_PX_AT_DEFAULT = 14   # checkbox / radio indicator width+height
_RADIO_RADIUS_PX_AT_DEFAULT = 8  # radio indicator border-radius
# DINO sidebar panel uses deliberately compact text (smaller than body).
# The widgets carry no inline font-size — these rules own it so the
# compact look is preserved but still scales with ui_font_pt.
_DINO_COMPACT_PX_AT_DEFAULT = 11  # threshold table, phrase list, buttons
_DINO_HINT_PX_AT_DEFAULT = 10     # italic hint under "Phrases for:"


def apply_theme_and_font(mw):
    font_size = mw.ui_font_pt
    style = soft_dark_stylesheet if mw.dark_mode else default_stylesheet
    # Appended rules win over the static sheet (same specificity,
    # later in cascade) — this is how the hardcoded px sizes in the
    # stylesheets are made to follow ui_font_pt without templating
    # the static strings.
    scale = font_size / FONT_PT_DEFAULT
    header_px = round(_HEADER_PX_AT_DEFAULT * scale)
    indicator_px = round(_INDICATOR_PX_AT_DEFAULT * scale)
    radio_radius_px = round(_RADIO_RADIUS_PX_AT_DEFAULT * scale)
    dino_px = round(_DINO_COMPACT_PX_AT_DEFAULT * scale)
    dino_hint_px = round(_DINO_HINT_PX_AT_DEFAULT * scale)
    combined_style = (
        f"{style}\n"
        f"QWidget {{ font-size: {font_size}pt; }}\n"
        f"QLabel.section-header {{ font-size: {header_px}px; }}\n"
        f"QCheckBox::indicator, QRadioButton::indicator {{"
        f" width: {indicator_px}px; height: {indicator_px}px; }}\n"
        f"QRadioButton::indicator {{ border-radius: {radio_radius_px}px; }}\n"
        f"ClassThresholdTable, ClassThresholdTable QDoubleSpinBox,"
        f" ClassThresholdTable QHeaderView::section,"
        f" PhraseEditorPanel QLabel, PhraseEditorPanel QListWidget,"
        f" PhraseEditorPanel QPushButton {{ font-size: {dino_px}px; }}\n"
        f"QLabel#dino_phrase_hint {{ font-size: {dino_hint_px}px; }}"
    )
    mw.setStyleSheet(combined_style)

    for widget in mw.findChildren(QWidget):
        font = widget.font()
        font.setPointSize(font_size)
        widget.setFont(font)

    mw.image_label.setFont(QFont("Arial", font_size))
    mw.image_label.set_ui_scale(font_size / FONT_PT_DEFAULT)
    mw.update()


def set_font_pt(mw, pt):
    mw.ui_font_pt = clamp_font_pt(pt)
    apply_theme_and_font(mw)
    save_ui_prefs(mw.ui_font_pt, mw.dark_mode)
    sync_font_menu(mw)


def step_font_pt(mw, delta):
    set_font_pt(mw, mw.ui_font_pt + delta)


def reset_font_pt(mw):
    set_font_pt(mw, FONT_PT_DEFAULT)


def change_font_size(mw, size):
    """Preset entry point — `size` is a name from `mw.font_sizes`."""
    set_font_pt(mw, mw.font_sizes[size])


def sync_font_menu(mw):
    """Check the preset action matching ui_font_pt, uncheck the rest.

    No preset is checked when the user stepped to an in-between size.
    """
    actions = getattr(mw, "_font_preset_actions", None)
    if not actions:
        return
    for name, action in actions.items():
        action.setChecked(mw.font_sizes[name] == mw.ui_font_pt)


def toggle_dark_mode(mw):
    mw.dark_mode = not mw.dark_mode
    apply_theme_and_font(mw)
    # Rebuild the image-list status badges for the new theme (#43). Routed
    # through the controller, not painted here.
    image_controller = getattr(mw, "image_controller", None)
    if image_controller is not None:
        image_controller.on_theme_changed()
    save_ui_prefs(mw.ui_font_pt, mw.dark_mode)
    mw.update_slice_list_colors()
    mw.update_class_list()
    mw.update_annotation_list()
    mw.repaint()


def apply_stylesheet(mw):
    mw.setStyleSheet(soft_dark_stylesheet if mw.dark_mode else default_stylesheet)


def update_ui_colors(mw):
    mw.update_annotation_list_colors()
    mw.update_slice_list_colors()
    mw.image_label.update()
