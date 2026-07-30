"""Global shortcuts and application-wide event filters for ImageAnnotator.

Two mechanisms live here, and which one a binding belongs to is a real
decision, not a style preference:

* :func:`install_shortcuts` registers **unconditional** bindings as ``QShortcut``
  with ``ApplicationShortcut`` context. Putting them in ``keyPressEvent`` didn't
  work because ``QTableWidget`` (the annotation table / DINO threshold table)
  and other focusable children consume the keys before they bubble up to the
  main window. These are all modified keys (Ctrl+Z, F2, ...) that no text field
  wants.

* :class:`ShortcutEventFilter` handles **conditional** bindings — bare digits
  and letters (issue #65). Those cannot be ``QShortcut``s: an
  ``ApplicationShortcut`` on ``3`` swallows the keystroke inside every
  ``QLineEdit`` in the app, so renaming a class to "Layer 3" or typing a DINO
  phrase would silently drop characters. An event filter is the only mechanism
  that can be *conditional* on where the focus currently is.

The filter is a small registry rather than another bespoke top-level filter,
which is what the ADR-015 follow-up note asked for:

    Future review modes should share filter or layer via strategy registry,
    not install multiple top-level filters.

:class:`DINOReviewEventFilter` keeps its own installation (it consumes
Enter/Escape, whose gating differs — it must stay inert unless temp
annotations are pending), but both now share the gate predicates in
:func:`no_modal_open` / :func:`focus_is_text_entry`.
"""

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from ..controllers.dino_controller import DINOReviewEventFilter
from ..core.constants import CLASS_KEY_LIMIT
from .input_gates import focus_is_text_entry, no_modal_open

# Letter -> canvas tool name. `None` means selection mode (no tool), the canvas
# default. Every value here is passed to ImageAnnotator.activate_tool, the
# single choke point for current_tool / sam_*_active / button check state
# (CLAUDE.md, "Tool Activation") -- the filter must never write those directly,
# or a SAM tool could end up active alongside a manual one.
TOOL_KEYS = {
    Qt.Key.Key_P: "polygon",
    Qt.Key.Key_R: "rectangle",
    Qt.Key.Key_B: "paint_brush",
    Qt.Key.Key_E: "eraser",
    Qt.Key.Key_K: "keypoint",
    Qt.Key.Key_V: None,
}

class ShortcutEventFilter(QObject):
    """Application-wide registry of conditional key bindings (issue #65).

    Holds ``{(key, modifiers): callable}`` plus a list of *gate* predicates.
    Every gate must pass before any binding fires, so adding a new global-key
    feature means registering a binding here rather than installing another
    top-level filter (the ADR-015 follow-up).

    A binding returning ``True`` consumes the event; ``False``/``None`` lets it
    through — which is what makes an out-of-range digit a silent no-op rather
    than an error.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bindings = {}
        self._gates = [no_modal_open, lambda: not focus_is_text_entry()]

    def register(self, key, handler, modifiers=Qt.KeyboardModifier.NoModifier):
        """Bind ``key`` (+ ``modifiers``) to ``handler``."""
        self._bindings[(key, modifiers)] = handler

    def add_gate(self, predicate):
        """Add a predicate that must pass before any binding fires."""
        self._gates.append(predicate)

    def gates_pass(self) -> bool:
        return all(gate() for gate in self._gates)

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        handler = self._bindings.get((event.key(), event.modifiers()))
        if handler is None:
            return False
        if not self.gates_pass():
            return False
        return bool(handler())


def _select_class_by_index(window, index):
    """Select the index-th class in the visible class list.

    Routed through ``ClassController.select_class`` so the list highlight and
    ``current_class`` move together, then ``on_class_selected`` for the rest of
    the selection side effects (tool enablement, the ADR-029 pose-class guards).
    Out of range is a no-op that does *not* consume the key.
    """
    if index >= window.class_list.count():
        return False
    window.class_controller.select_class(index)
    item = window.class_list.item(index)
    if item is not None:
        window.on_class_selected(item)
    return True


def _activate_tool_by_key(window, tool_name):
    """Activate ``tool_name``, or toggle back to selection mode if it is already
    active — matching the click-to-toggle-off behaviour of the sidebar buttons.
    """
    if tool_name is not None and window.image_label.current_tool == tool_name:
        tool_name = None
    if not window.image_label.check_unsaved_changes():
        return True  # the user declined the save prompt; still consume the key
    window.activate_tool(tool_name)
    return True


def install_shortcuts(window):
    """Register global keyboard shortcuts (see the module docstring for why
    these are QShortcuts and the issue-#65 bindings are not)."""
    window._snake_shortcut = QShortcut(QKeySequence("F2"), window)
    window._snake_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._snake_shortcut.activated.connect(window.launch_snake_game)

    # Undo / redo of annotation edits (ADR-026). Ctrl+Z, plus Ctrl+Y and
    # Ctrl+Shift+Z as cross-platform redo aliases.
    ac = window.annotation_controller
    window._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, window)
    window._undo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._undo_shortcut.activated.connect(ac.undo)

    window._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, window)
    window._redo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._redo_shortcut.activated.connect(ac.redo)

    window._redo_shortcut_alt = QShortcut(QKeySequence("Ctrl+Shift+Z"), window)
    window._redo_shortcut_alt.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._redo_shortcut_alt.activated.connect(ac.redo)


def build_shortcut_filter(window):
    """Build the conditional-binding registry: class digits and tool letters
    (issue #65), plus Ctrl+C / Ctrl+V for the annotation clipboard (issue #66).

    Split out from :func:`install_event_filters` so tests can build and drive
    the filter without installing it on the QApplication.

    Copy/paste belongs here rather than in :func:`install_shortcuts` for the
    same reason as the bare keys: Ctrl+C inside a class-rename field must copy
    *text*, not annotations. The gate predicates already express exactly that,
    and an ApplicationShortcut could not.
    """
    filt = ShortcutEventFilter(window)

    for offset in range(CLASS_KEY_LIMIT):
        key = getattr(Qt.Key, f"Key_{offset + 1}")
        filt.register(key, (lambda i: lambda: _select_class_by_index(window, i))(offset))

    for key, tool_name in TOOL_KEYS.items():
        filt.register(
            key, (lambda t: lambda: _activate_tool_by_key(window, t))(tool_name)
        )

    clipboard = window.clipboard_controller
    filt.register(
        Qt.Key.Key_C, clipboard.copy_selection, Qt.KeyboardModifier.ControlModifier
    )
    filt.register(Qt.Key.Key_V, clipboard.paste, Qt.KeyboardModifier.ControlModifier)

    return filt


def install_event_filters(window):
    """Install application-wide event filters.

    Two of them: the DINO review filter (Enter/Escape for pending temp
    annotations, ADR-015) and the issue-#65 shortcut registry. They stay
    separate objects because their gating differs — the DINO filter must also
    check that temp annotations are actually pending — but they share the gate
    predicates in this module.
    """
    window._dino_review_filter = DINOReviewEventFilter(window)
    QApplication.instance().installEventFilter(window._dino_review_filter)

    window._shortcut_filter = build_shortcut_filter(window)
    QApplication.instance().installEventFilter(window._shortcut_filter)
