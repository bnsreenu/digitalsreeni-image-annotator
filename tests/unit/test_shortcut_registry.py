"""Shortcut registry and its gate predicates (issue #65, ADR-043).

The gates are the whole safety story of this feature. Binding bare digits
application-wide is only acceptable because the filter can ask *where is the
focus right now* and stand down — a plain ``QShortcut`` cannot, which is why
digits could never be registered that way.

These tests drive :class:`ShortcutEventFilter` directly (built via
``build_shortcut_filter`` but not installed on the QApplication), so no real
main window or event loop is needed for the registry logic itself.
"""

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSpinBox,
    QTextEdit,
    QWidget,
)

from src.digitalsreeni_image_annotator.ui.input_gates import (
    focus_is_text_entry,
    no_modal_open,
    widget_is_text_entry,
)
from src.digitalsreeni_image_annotator.ui.shortcuts import (
    ShortcutEventFilter,
    build_shortcut_filter,
)


def _key_event(key, modifiers=Qt.KeyboardModifier.NoModifier):
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


# --- gate predicates -------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox],
    ids=["line", "text", "plain", "spin", "doublespin"],
)
def test_text_widgets_are_classified_as_text_entry(qtbot, factory):
    widget = factory()
    qtbot.addWidget(widget)
    assert widget_is_text_entry(widget) is True


def test_a_spin_boxs_embedded_editor_counts_too(qtbot):
    """Some styles focus the QLineEdit *inside* the spin box rather than the
    box itself; both must classify the same way or the gate is style-dependent.
    """
    spin = QSpinBox()
    qtbot.addWidget(spin)
    editor = spin.findChild(QLineEdit)
    assert editor is not None
    assert widget_is_text_entry(editor) is True


def test_editable_combo_counts_as_text_entry(qtbot):
    combo = QComboBox()
    combo.setEditable(True)
    qtbot.addWidget(combo)
    assert widget_is_text_entry(combo) is True
    assert widget_is_text_entry(combo.lineEdit()) is True


def test_non_editable_combo_does_not_count_as_text_entry(qtbot):
    """A dropdown takes no typed text, so gating on it would disable the
    bindings for no reason."""
    combo = QComboBox()
    combo.addItems(["a", "b"])
    qtbot.addWidget(combo)
    assert widget_is_text_entry(combo) is False


def test_a_list_is_not_text_entry(qtbot):
    listw = QListWidget()
    listw.addItem(QListWidgetItem("cell"))
    qtbot.addWidget(listw)
    assert widget_is_text_entry(listw) is False


def test_no_focus_is_not_text_entry():
    assert widget_is_text_entry(None) is False


def test_focus_lookup_delegates_to_the_classifier(qtbot, monkeypatch):
    """``focus_is_text_entry`` is only the QApplication lookup; the decision is
    ``widget_is_text_entry``'s. Asserted by driving the lookup, because real
    focus is unreliable under the offscreen platform."""
    from PyQt6.QtWidgets import QApplication

    field = QLineEdit()
    qtbot.addWidget(field)
    monkeypatch.setattr(QApplication.instance(), "focusWidget", lambda: field)
    assert focus_is_text_entry() is True

    listw = QListWidget()
    qtbot.addWidget(listw)
    monkeypatch.setattr(QApplication.instance(), "focusWidget", lambda: listw)
    assert focus_is_text_entry() is False


def test_no_modal_open_is_true_without_a_dialog(qtbot):
    assert no_modal_open() is True


# --- registry mechanics ----------------------------------------------------


def test_registered_binding_fires_and_consumes(qtbot):
    filt = ShortcutEventFilter()
    fired = []
    filt.register(Qt.Key.Key_1, lambda: fired.append(1) or True)

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_1)) is True
    assert fired == [1]


def test_unregistered_key_passes_through(qtbot):
    filt = ShortcutEventFilter()
    assert filt.eventFilter(None, _key_event(Qt.Key.Key_Z)) is False


def test_binding_returning_false_does_not_consume(qtbot):
    """This is what makes an out-of-range digit a silent no-op: the handler
    declines, so the key continues to whatever else wants it."""
    filt = ShortcutEventFilter()
    filt.register(Qt.Key.Key_9, lambda: False)
    assert filt.eventFilter(None, _key_event(Qt.Key.Key_9)) is False


def test_modifiers_are_part_of_the_binding_key(qtbot):
    filt = ShortcutEventFilter()
    fired = []
    filt.register(
        Qt.Key.Key_C, lambda: fired.append("ctrl") or True,
        Qt.KeyboardModifier.ControlModifier,
    )

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_C)) is False
    assert fired == []
    assert filt.eventFilter(
        None, _key_event(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    ) is True
    assert fired == ["ctrl"]


def test_non_keypress_events_are_ignored(qtbot):
    filt = ShortcutEventFilter()
    filt.register(Qt.Key.Key_1, lambda: True)
    release = QKeyEvent(
        QEvent.Type.KeyRelease, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier
    )
    assert filt.eventFilter(None, release) is False


def test_a_failing_gate_suppresses_every_binding(qtbot):
    filt = ShortcutEventFilter()
    fired = []
    filt.register(Qt.Key.Key_1, lambda: fired.append(1) or True)
    filt.add_gate(lambda: False)

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_1)) is False
    assert fired == []


def test_text_focus_suppresses_a_digit_binding(qtbot, monkeypatch):
    """The headline regression: typing "3" into a rename field must reach the
    field, not switch class."""
    from PyQt6.QtWidgets import QApplication

    field = QLineEdit()
    qtbot.addWidget(field)
    monkeypatch.setattr(QApplication.instance(), "focusWidget", lambda: field)

    filt = ShortcutEventFilter()
    fired = []
    filt.register(Qt.Key.Key_3, lambda: fired.append(3) or True)

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_3)) is False
    assert fired == []


def test_a_modal_dialog_suppresses_every_binding(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QApplication, QDialog

    dialog = QDialog()
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QApplication.instance(), "activeModalWidget", lambda: dialog)

    filt = ShortcutEventFilter()
    fired = []
    filt.register(Qt.Key.Key_1, lambda: fired.append(1) or True)

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_1)) is False
    assert fired == []
    assert no_modal_open() is False


# --- window bindings -------------------------------------------------------


class _FakeImageLabel:
    def __init__(self):
        self.current_tool = None
        self.unsaved_ok = True

    def check_unsaved_changes(self):
        return self.unsaved_ok


class _FakeClassController:
    def __init__(self, window):
        self.window = window

    def select_class(self, index):
        self.window.current_class = self.window.class_list.item(index).text()


class _FakeClipboardController:
    def __init__(self):
        self.copied = 0
        self.pasted = 0

    def copy_selection(self):
        self.copied += 1
        return True

    def paste(self):
        self.pasted += 1
        return True


class _FakeWindow(QWidget):
    """Just enough of ImageAnnotator for the bindings: a class list, the
    activate_tool choke point, and the controllers they route through."""

    def __init__(self, classes=("cell", "nucleus", "debris")):
        super().__init__()
        self.class_list = QListWidget(self)
        for name in classes:
            self.class_list.addItem(QListWidgetItem(name))
        self.current_class = None
        self.activated = []
        self.image_label = _FakeImageLabel()
        self.class_controller = _FakeClassController(self)
        self.clipboard_controller = _FakeClipboardController()
        self.selected_items = []

    def on_class_selected(self, item):
        self.selected_items.append(item.text())

    def activate_tool(self, tool_name):
        self.activated.append(tool_name)
        self.image_label.current_tool = tool_name


@pytest.fixture
def window(qtbot):
    win = _FakeWindow()
    qtbot.addWidget(win)
    return win


@pytest.mark.parametrize(
    "key,expected",
    [
        (Qt.Key.Key_1, "cell"),
        (Qt.Key.Key_2, "nucleus"),
        (Qt.Key.Key_3, "debris"),
    ],
)
def test_digit_selects_the_matching_class(window, key, expected):
    filt = build_shortcut_filter(window)
    assert filt.eventFilter(None, _key_event(key)) is True
    assert window.current_class == expected
    assert window.selected_items[-1] == expected


def test_out_of_range_digit_is_a_no_op(window):
    """Three classes exist; pressing 7 must not raise and must not consume."""
    filt = build_shortcut_filter(window)
    assert filt.eventFilter(None, _key_event(Qt.Key.Key_7)) is False
    assert window.current_class is None


@pytest.mark.parametrize(
    "key,tool",
    [
        (Qt.Key.Key_P, "polygon"),
        (Qt.Key.Key_R, "rectangle"),
        (Qt.Key.Key_B, "paint_brush"),
        (Qt.Key.Key_E, "eraser"),
        (Qt.Key.Key_K, "keypoint"),
    ],
)
def test_letter_activates_the_matching_tool(window, key, tool):
    filt = build_shortcut_filter(window)
    assert filt.eventFilter(None, _key_event(key)) is True
    assert window.activated == [tool]


def test_v_returns_to_selection_mode(window):
    filt = build_shortcut_filter(window)
    window.image_label.current_tool = "polygon"
    filt.eventFilter(None, _key_event(Qt.Key.Key_V))
    assert window.activated == [None]


def test_pressing_the_active_tool_key_toggles_it_off(window):
    """Matches the click-to-toggle-off behaviour of the sidebar buttons."""
    filt = build_shortcut_filter(window)
    filt.eventFilter(None, _key_event(Qt.Key.Key_P))
    filt.eventFilter(None, _key_event(Qt.Key.Key_P))
    assert window.activated == ["polygon", None]


def test_tool_switch_is_blocked_by_a_declined_save_prompt(window):
    """An unsaved stroke the user chose to keep must not be dropped by a
    keystroke any more than by a button click."""
    window.image_label.unsaved_ok = False
    filt = build_shortcut_filter(window)
    assert filt.eventFilter(None, _key_event(Qt.Key.Key_P)) is True
    assert window.activated == []


def test_tool_activation_never_writes_current_tool_directly(window):
    """The binding must go through activate_tool -- the single choke point that
    keeps a SAM tool from being active alongside a manual one (CLAUDE.md)."""
    filt = build_shortcut_filter(window)
    window.activate_tool = lambda name: window.activated.append(name)  # no side effect
    filt.eventFilter(None, _key_event(Qt.Key.Key_R))
    assert window.image_label.current_tool is None, (
        "current_tool changed without going through activate_tool"
    )


# --- clipboard bindings (issue #66) ---------------------------------------


def test_ctrl_c_and_ctrl_v_reach_the_clipboard(window):
    filt = build_shortcut_filter(window)
    ctrl = Qt.KeyboardModifier.ControlModifier

    assert filt.eventFilter(None, _key_event(Qt.Key.Key_C, ctrl)) is True
    assert filt.eventFilter(None, _key_event(Qt.Key.Key_V, ctrl)) is True
    assert window.clipboard_controller.copied == 1
    assert window.clipboard_controller.pasted == 1


def test_bare_v_is_the_tool_binding_not_paste(window):
    """V and Ctrl+V are different bindings; the modifier is part of the key."""
    filt = build_shortcut_filter(window)
    filt.eventFilter(None, _key_event(Qt.Key.Key_V))
    assert window.activated == [None]
    assert window.clipboard_controller.pasted == 0


def test_ctrl_c_in_a_text_field_is_left_alone(window, qtbot, monkeypatch):
    """Copying text out of a class-rename field must stay normal text copy."""
    from PyQt6.QtWidgets import QApplication

    field = QLineEdit()
    qtbot.addWidget(field)
    monkeypatch.setattr(QApplication.instance(), "focusWidget", lambda: field)

    filt = build_shortcut_filter(window)
    consumed = filt.eventFilter(
        None, _key_event(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    )

    assert consumed is False
    assert window.clipboard_controller.copied == 0
