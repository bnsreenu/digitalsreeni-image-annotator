"""Image-list item delegate that paints the review score badge (issue #71).

The image list is a ``QListWidget``, so "add a score column" means painting
one. It cannot go into the item text for the same reason the class-list
shortcut badge cannot: **the item text IS the file name** throughout the app —
DINO batch navigation matches on it, COCO import reconciliation matches on it,
``findItems(name, MatchExactly)`` re-selects on it, and the
``all_images[i]`` ↔ ``image_list.item(i)`` positional invariant is maintained
alongside it (ADR-035). Appending " (4.2)" would break every one of those, at
runtime only.

So the badge lives in the paint pass, where it is visual and inert.
"""

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QStyledItemDelegate


class ImageScoreDelegate(QStyledItemDelegate):
    """Right-aligned review score on each image row, when one exists."""

    _BADGE_ALPHA = 190

    def __init__(self, parent, score_lookup):
        """``score_lookup(file_name) -> float | None``.

        A callable rather than a dict so the delegate always reads the live
        scores; a snapshot would go stale the moment a run finished.
        """
        super().__init__(parent)
        self._score_lookup = score_lookup

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        score = self._score_lookup(index.data(Qt.ItemDataRole.DisplayRole))
        if score is None:
            return

        painter.save()
        # Derived from the row's own text colour rather than a literal, so it
        # reads in both themes (CLAUDE.md, "No Hardcoded Colors Rule").
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
            f"{score:.1f}",
        )
        painter.restore()
