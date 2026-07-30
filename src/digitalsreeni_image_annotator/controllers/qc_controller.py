"""Annotation QC audit orchestration (issue #70).

The rules themselves live in :mod:`core.annotation_qc`, Qt-free so the headless
CLI can run them as a CI gate (issue #76). This controller is the GUI adapter:
it gathers the inputs the engine needs (annotations, image sizes, class names),
shows the dialog, and applies repairs through the undo choke point.
"""

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QMessageBox

from ..core import annotation_qc
from ..core.logging_config import get_logger
from ..core.slice_cache import slice_names

logger = get_logger(__name__)


class QCController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.config = annotation_qc.QCConfig()

    def collect_image_sizes(self):
        """``{image_or_slice_name: (width, height)}`` for the whole project.

        Sizes come from ``image_shapes`` where the project recorded them.
        Slices of a multi-dimensional image inherit their stack's H/W — they
        are the same raster, and enumerating them here is what stops the audit
        silently skipping every slice, the trap
        ``_collect_dino_batch_work_items`` exists to avoid.

        An image whose size cannot be resolved is simply omitted; the bounds
        rules skip it and the rest of the audit still runs. A partial answer
        beats refusing to answer.
        """
        sizes = {}
        for info in self.mw.all_images:
            file_name = info.get("file_name")
            shape = self.mw.image_shapes.get(file_name)
            size = _hw_from_shape(shape)
            if size:
                sizes[file_name] = size

        for base_name, slices in self.mw.image_slices.items():
            size = None
            for file_name, recorded in self.mw.image_shapes.items():
                if file_name.rsplit(".", 1)[0] == base_name:
                    size = _hw_from_shape(recorded)
                    break
            if size is None:
                continue
            for name in slice_names(slices):
                sizes[name] = size

        # The image on screen is authoritative for its own size, whatever the
        # project recorded.
        current = self.mw.current_slice or self.mw.image_file_name
        pixmap = self.mw.image_label.original_pixmap
        if current and pixmap is not None:
            sizes[current] = (pixmap.width(), pixmap.height())
        return sizes

    def run_audit(self):
        """Run the rules over the project and show the findings."""
        self.mw.save_current_annotations()
        if not self.mw.all_annotations:
            QMessageBox.information(
                self.mw, "Check Annotations", "There are no annotations to check."
            )
            return

        try:
            findings = annotation_qc.run_audit(
                self.mw.all_annotations,
                image_sizes=self.collect_image_sizes(),
                class_names=list(self.mw.image_label.class_colors.keys()),
                config=self.config,
            )
        except Exception:
            logger.exception("Annotation QC audit failed")
            QMessageBox.critical(
                self.mw,
                "Check Annotations",
                "The audit failed. See the log for details.",
            )
            return

        if not findings:
            QMessageBox.information(
                self.mw,
                "Check Annotations",
                "No problems found. The project passes every rule.",
            )
            return

        from ..dialogs.annotation_qc_dialog import AnnotationQCDialog

        AnnotationQCDialog(self.mw, findings).exec()

    def fix_findings(self, findings):
        """Repair every fixable finding. Returns the number actually changed.

        **One history entry per image, not one for the batch.**
        ``AnnotationHistory`` is keyed by image, and a finding can name any
        image in the project, not just the one on screen. A single keyless
        ``record_history()`` would snapshot only the current image — so repairs
        to the other 49 images of a 50-image project would be permanent and
        un-undoable while the UI claimed otherwise.

        The consequence is that undoing a cross-image sweep takes one Ctrl+Z
        per affected image. That is the honest cost of a per-image undo model
        (ADR-026); claiming otherwise was the bug.

        Returns ``(repaired_count, image_names)`` — the **names**, not a count,
        so the caller can tell the user which images to visit. Undo is keyed by
        image and ``undo()`` acts on whichever image is current, so "press
        Ctrl+Z N times" is not an instruction anyone can follow.
        """
        if not findings:
            return 0, []
        sizes = self.collect_image_sizes()
        current = self.mw.current_slice or self.mw.image_file_name

        snapshotted = set()
        repaired = 0
        for finding in findings:
            annotation = self._resolve(finding)
            if annotation is None:
                continue
            # Snapshot this image's pre-repair state the first time we touch it,
            # and before the mutation below.
            if finding.image not in snapshotted:
                self.mw.annotation_controller.record_history(finding.image)
                snapshotted.add(finding.image)
            width, height = sizes.get(finding.image, (None, None))
            try:
                if annotation_qc.apply_fix(annotation, finding.rule, width, height):
                    repaired += 1
            except Exception:
                logger.exception(
                    "QC repair failed for %s on %s", finding.rule, finding.image
                )

        # The on-screen image's working copy is a deep copy of all_annotations,
        # so a repair written into all_annotations has to be reloaded to show
        # up on the canvas.
        if any(f.image == current for f in findings):
            self.mw.load_image_annotations()
            self.mw.update_annotation_list()
        self.mw.image_label.update()
        self.mw.auto_save()
        logger.info(
            "QC repaired %d finding(s) across %d image(s)", repaired, len(snapshotted)
        )
        return repaired, sorted(snapshotted)

    def _resolve(self, finding):
        """The live annotation a finding refers to, or None.

        Repairs are written into ``all_annotations`` — the project-level store —
        because a finding can name any image, not just the one on screen.
        """
        if not finding.image or not finding.class_name:
            return None
        by_class = self.mw.all_annotations.get(finding.image) or {}
        for annotation in by_class.get(finding.class_name, []):
            if annotation.get("number") == finding.annotation_number:
                return annotation
        return None


def _hw_from_shape(shape):
    """``(width, height)`` from a recorded image shape, or None.

    Shapes are numpy-ordered ``(..., H, W)``; the last two axes are the raster
    for both a plain 2D image and an N-dimensional stack.
    """
    if not shape or len(shape) < 2:
        return None
    try:
        height, width = int(shape[-2]), int(shape[-1])
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None
