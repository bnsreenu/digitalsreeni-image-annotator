"""Shared gate predicates for application-wide key event filters (issue #65).

An application-wide event filter that reacts to keys has to answer the same two
questions every time: *is a modal dialog up?* and *is the user typing?* Getting
either wrong is not a cosmetic bug — it means a keystroke meant for a text field
is stolen, or a canvas action fires under a dialog the user is looking at.

``DINOReviewEventFilter`` (ADR-015) answered them inline. The issue-#65
shortcut registry needs the same answers with a wider notion of "typing"
(spin boxes and editable combos matter once bare digits are bound). Rather than
let two filters drift apart on a safety check, both import from here.

This module lives under ``ui/`` but is imported by ``controllers/`` too; it has
no dependency on either, which is what keeps that import cycle-free.
"""

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)


def no_modal_open() -> bool:
    """True when no modal dialog is up.

    A modal owns the keyboard completely; firing a canvas binding underneath it
    would act on state the user cannot see, and would break the dialog's own
    default-button handling.
    """
    app = QApplication.instance()
    return app is not None and app.activeModalWidget() is None


def widget_is_text_entry(widget) -> bool:
    """True when ``widget`` accepts typed text.

    Split from :func:`focus_is_text_entry` so the *classification* can be
    tested directly: which widget currently holds focus is a property of the
    windowing system (and is unreliable under the offscreen platform used in
    CI), whereas "is a QSpinBox a text entry" is a decision this module owns
    and must get right.

    Covers the single- and multi-line editors, spin boxes, and **editable**
    combo boxes. A non-editable combo takes no typed text, so treating it as
    one would needlessly disable the bindings whenever a dropdown has focus.
    """
    if widget is None:
        return False
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return True
    if isinstance(widget, QComboBox) and widget.isEditable():
        return True
    # Spin boxes and editable combos focus their *embedded* editor on some
    # styles. That child is a QLineEdit and is already caught above; this
    # covers any other editor child a style might focus instead, and the
    # non-editable-combo case correctly returns False because its line edit
    # does not exist.
    parent = widget.parent()
    if isinstance(parent, QAbstractSpinBox):
        return True
    return isinstance(parent, QComboBox) and parent.isEditable()


def focus_is_text_entry() -> bool:
    """True when the widget that currently has focus accepts typed text.

    This is the predicate that makes bare-key bindings safe at all.
    """
    app = QApplication.instance()
    if app is None:
        return False
    return widget_is_text_entry(app.focusWidget())
