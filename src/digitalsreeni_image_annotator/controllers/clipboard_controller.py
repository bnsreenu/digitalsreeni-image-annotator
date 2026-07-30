"""Annotation clipboard: copy and paste across images, slices and frames (#66).

The same object appears on dozens of consecutive Z-slices or video frames,
barely moved. Before this, it had to be redrawn from scratch every time — there
was no way to reuse a shape you had already made, anywhere.

The clipboard is **app-level**, not per-image: it survives switching image,
slice, frame and project, because "copy here, paste 40 slices later" is the
whole point. It is deliberately *not* the system clipboard — annotations are
rich dicts with class bindings and schema constraints, and round-tripping them
through text would lose the class-mapping decision this controller exists to
make.

Three rules govern the paste, and each exists because of a specific way the
naive version goes wrong:

* **Deep copy on the way in and on the way out.** ``image_label.annotations``
  is itself a deep copy of ``all_annotations`` and PyQt round-trips ``UserRole``
  dicts as copies, so value-equality is the only stable identity (ADR-022).
  Holding a reference would make the pasted shape an alias of the source —
  editing one would silently edit the other.
* **Clamp, don't clip.** The target image may be smaller than the source. The
  per-coordinate clamp preserves vertex count and ordering (ADR-024); a
  shapely clip would be geometrically prettier and would also change the vertex
  count out from under a shape the user is about to nudge.
* **A pose only travels to a class with the same schema.** K is locked once
  instances exist (ADR-029); pasting a 17-point pose onto a 5-point class does
  not produce a slightly-wrong instance, it produces a corrupt one. Mismatches
  are reported and skipped rather than coerced.
"""

import copy

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from ..core.keypoint_schema import schema_k
from ..core.logging_config import get_logger
from ..utils import clamp_bbox, clamp_keypoints, clamp_segmentation

logger = get_logger(__name__)

# Per-image bookkeeping that must not travel with a copied annotation. `number`
# is assigned fresh by add_annotation_to_list against the target image's
# existing annotations; carrying the source value would collide.
_VOLATILE_KEYS = ("number",)

# Sentinel returned by the class-resolution dialog when the user chooses to
# create the missing class rather than map it onto an existing one.
CREATE_NEW = object()


class ClassMappingDialog(QDialog):
    """Asks what to do about a pasted class that the target project lacks.

    Shown once per *distinct* missing class name, not once per annotation —
    pasting 200 cells into a project without a "cell" class must ask once.
    """

    def __init__(self, class_name, existing_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste: unknown class")
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"The pasted annotations use the class '{class_name}', which does "
                "not exist here.\nCreate it, or map them onto an existing class?"
            )
        )
        self.combo = QComboBox()
        self.combo.addItem(f"Create '{class_name}'", CREATE_NEW)
        for name in existing_classes:
            self.combo.addItem(f"Map onto '{name}'", name)
        layout.addWidget(self.combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def chosen(self):
        """:data:`CREATE_NEW` or an existing class name."""
        return self.combo.currentData()


class ClipboardController(QObject):
    """Owns the in-app annotation clipboard and the paste workflow."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self._entries = []

    # --- introspection (used by tests and by any future paste-menu item) ---

    def has_content(self) -> bool:
        return bool(self._entries)

    def entry_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries = []

    # --- copy ---

    def copy_selection(self) -> bool:
        """Copy the current canvas/list selection into the clipboard.

        Returns True when the key was consumed. An empty selection returns
        False so Ctrl+C keeps whatever other meaning it has in that context.
        """
        selection = list(self.mw.image_label.highlighted_annotations)
        if not selection:
            return False

        entries = []
        for annotation in selection:
            entry = copy.deepcopy(annotation)
            for key in _VOLATILE_KEYS:
                entry.pop(key, None)
            entries.append(entry)
        self._entries = entries

        logger.debug("Copied %d annotation(s) to the clipboard", len(entries))
        return True

    # --- paste ---

    def paste(self) -> bool:
        """Paste the clipboard into the current image at the original coordinates.

        Returns True when the key was consumed.
        """
        if not self._entries:
            return False
        pixmap = self.mw.image_label.original_pixmap
        if pixmap is None:
            return False

        mapping = self._resolve_classes()
        if mapping is None:
            return True  # user cancelled at the class dialog; key still consumed

        width, height = pixmap.width(), pixmap.height()
        ac = self.mw.annotation_controller

        # One history entry for the whole paste, recorded before any mutation,
        # so a single Ctrl+Z undoes it however many annotations it contained
        # (ADR-026).
        ac.record_history()

        pasted, skipped = [], []
        for entry in self._entries:
            target_class = mapping.get(entry.get("category_name"))
            if target_class is None:
                continue
            annotation = self._prepare(entry, target_class, width, height)
            if annotation is None:
                skipped.append(entry.get("category_name"))
                continue
            self.mw.image_label.annotations.setdefault(target_class, []).append(
                annotation
            )
            ac.add_annotation_to_list(annotation)
            pasted.append(annotation)

        if not pasted:
            self._warn_all_skipped(skipped)
            return True

        ac.save_current_annotations()
        self.mw.update_slice_list_colors()
        # The pasted shapes become the selection so they can be nudged straight
        # away with the #40 handles.
        ac.apply_canvas_selection(pasted, "replace")
        self.mw.image_label.update()
        self.mw.auto_save()

        if skipped:
            self._warn_skipped(skipped)
        logger.debug("Pasted %d annotation(s), skipped %d", len(pasted), len(skipped))
        return True

    # --- internals ---

    def _resolve_classes(self):
        """Map every clipboard class name onto a class in the target project.

        Asks once per distinct missing name. Returns ``None`` if the user
        cancels, in which case nothing at all is pasted — a partial paste after
        a cancel would be worse than none.
        """
        mapping = {}
        for entry in self._entries:
            source = entry.get("category_name")
            if source in mapping:
                continue
            if source in self.mw.class_mapping:
                mapping[source] = source
                continue

            existing = [
                name
                for name in self.mw.class_mapping
                if not name.startswith("Temp-")
            ]
            dialog = ClassMappingDialog(source, existing, self.mw)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            choice = dialog.chosen()
            if choice is CREATE_NEW:
                self.mw.add_class(
                    source, self.mw.image_label.class_colors.get(source)
                )
                mapping[source] = source
            else:
                mapping[source] = choice
        return mapping

    def _prepare(self, entry, target_class, width, height):
        """Build the annotation to insert, or ``None`` to skip it.

        Skipping happens for exactly one reason: a pose whose schema does not
        match the target class's. Everything else is representable.
        """
        annotation = copy.deepcopy(entry)
        annotation["category_name"] = target_class
        annotation["category_id"] = self.mw.class_mapping.get(target_class, 0)

        if "keypoints" in annotation:
            if not self._pose_fits(annotation, target_class):
                return None
            annotation["keypoints"] = clamp_keypoints(
                annotation["keypoints"], width, height
            )
            if annotation.get("bbox") is not None:
                annotation["bbox"] = clamp_bbox(annotation["bbox"], width, height)
            # A pose carries no segmentation key; its absence is the
            # discriminator that routes area, Detail-% and rendering (ADR-029).
            annotation.pop("segmentation", None)
            annotation.pop("segmentation_raw", None)
            return annotation

        if annotation.get("segmentation"):
            annotation["segmentation"] = clamp_segmentation(
                annotation["segmentation"], width, height
            )
            # Keep the raw copy so the Detail-% spinbox stays reversible on the
            # pasted shape (ADR-025) -- clamped the same way, or a 100 % reset
            # would push coordinates back out of bounds.
            if annotation.get("segmentation_raw"):
                annotation["segmentation_raw"] = clamp_segmentation(
                    annotation["segmentation_raw"], width, height
                )
        if annotation.get("bbox") is not None:
            annotation["bbox"] = clamp_bbox(annotation["bbox"], width, height)
        return annotation

    def _pose_fits(self, annotation, target_class):
        """True when a pose instance can legally join ``target_class``.

        Requires an exact K match. A target class with no schema at all cannot
        take the instance either: defining the schema is a separate, deliberate
        act (ADR-029), and inventing one here would lock K by accident.
        """
        target_schema = self.mw.keypoint_schemas.get(target_class)
        if target_schema is None:
            return False
        return schema_k(target_schema) == len(annotation.get("keypoints", [])) // 3

    def _warn_skipped(self, skipped):
        names = ", ".join(sorted(set(n for n in skipped if n)))
        QMessageBox.warning(
            self.mw,
            "Some annotations were not pasted",
            f"{len(skipped)} pose instance(s) from '{names}' were skipped: the "
            "target class has no keypoint schema, or a different number of "
            "keypoints. Define a matching schema on the target class first.",
        )

    def _warn_all_skipped(self, skipped):
        if skipped:
            self._warn_skipped(skipped)
        else:
            QMessageBox.information(
                self.mw, "Nothing pasted", "The clipboard produced no annotations."
            )
