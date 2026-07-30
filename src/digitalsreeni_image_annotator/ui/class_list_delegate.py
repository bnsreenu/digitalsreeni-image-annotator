"""Class-list item delegate that paints the digit shortcut badge (issue #65).

The digit hotkeys are useless if nobody knows they exist, so the first nine
classes show their number. It is painted rather than appended to the item text
because **the item text IS the class name** throughout the app:
``findItems(name, MatchExactly)`` re-selects the current class,
``item.text().startswith("Temp-")`` drives the whole review workflow,
``item.text()[5:]`` derives the permanent class name on accept, and rename
reads it back. Decorating the text would break every one of those in a way that
only shows up at runtime.

So the badge lives in the paint pass, where it is purely visual and cannot leak
into any of the string logic.
"""

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QStyledItemDelegate

from ..core.constants import CLASS_KEY_LIMIT


class ClassShortcutDelegate(QStyledItemDelegate):
    """Draws a right-aligned ``1``..``9`` badge on the first nine class rows."""

    # Badge colour is derived from the row's own text colour at reduced alpha
    # rather than a literal, so it reads correctly in both themes without a
    # hardcoded value (CLAUDE.md, "No Hardcoded Colors Rule").
    _BADGE_ALPHA = 140

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if index.row() >= CLASS_KEY_LIMIT:
            return

        painter.save()
        colour = QColor(option.palette.text().color())
        colour.setAlpha(self._BADGE_ALPHA)
        painter.setPen(colour)

        font = QFont(option.font)
        font.setBold(True)
        painter.setFont(font)

        rect = QRect(option.rect)
        rect.setRight(rect.right() - 6)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            str(index.row() + 1),
        )
        painter.restore()
