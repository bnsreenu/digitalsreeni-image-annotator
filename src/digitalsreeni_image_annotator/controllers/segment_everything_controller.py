"""Segment Everything: unprompted SAM proposals into the review overlay (#69).

Every SAM-assisted annotation was one object at a time: draw a box or click
points, wait, accept. On a dense image — cells, particles, grains — that is
dozens of identical prompt-and-wait cycles for a result SAM can produce in one
pass. It could segment everything without any prompt at all; the app never
asked.

**No second review mechanic.** The proposals land in the *existing*
``temp_annotations`` overlay under a ``Temp-Auto`` class, tagged
``source: "sam-everything"``, and Enter/Escape go through the same
application-wide filter DINO and SAM 3 already use. ADR-015 explicitly warns
against layering a second top-level review mode, and a dedicated hover-preview
mode was considered and rejected on exactly that basis.

What is new is only the *assignment* step: an unprompted pass has no idea what
the objects are, so each candidate needs a class before it can be committed.
Clicking one assigns the active class, which is what makes the digit hotkeys
from issue #65 the difference between "many clicks" and "fast".
"""

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..core import mask_filters
from ..core.constants import default_class_color
from ..core.logging_config import get_logger
from ..inference.sam_utils import InferenceBusyError

logger = get_logger(__name__)

# The class the proposals live under until the user assigns real ones. Named
# with the Temp- prefix so the existing review machinery recognises it, and
# popped on both accept and reject so no orphan survives -- leftover Temp-*
# classes are exactly what the #63 rename guard had to work around.
TEMP_AUTO_CLASS = "Temp-Auto"
SOURCE = mask_filters.SAM_EVERYTHING_SOURCE


class SegmentEverythingController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.min_area = mask_filters.DEFAULT_MIN_AREA
        self.max_area_fraction = mask_filters.DEFAULT_MAX_AREA_FRACTION
        self.max_candidates = mask_filters.DEFAULT_MAX_CANDIDATES
        self.overlap_iou = mask_filters.DEFAULT_OVERLAP_IOU

    # --- run ---

    def run(self):
        """Segment the current image with no prompt and stage the proposals."""
        if self.mw.current_image is None:
            QMessageBox.warning(
                self.mw, "Segment Everything", "Open an image first."
            )
            return
        if not self.mw.sam_utils.current_sam_model:
            QMessageBox.warning(
                self.mw,
                "Segment Everything",
                "Pick a SAM model first (Annotation → SAM model selector).",
            )
            return
        if self.mw.image_label.temp_annotations:
            QMessageBox.warning(
                self.mw,
                "Segment Everything",
                "There are pending proposals to review. Accept them with Enter "
                "or discard them with Escape first.",
            )
            return

        progress = QProgressDialog(
            "Segmenting everything…", "Cancel", 0, 0, self.mw
        )
        progress.setWindowTitle("Segment Everything")
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        cancelled = False
        try:
            proposals = self.mw.sam_utils.apply_sam_everything(self.mw.current_image)
        except InferenceBusyError:
            QMessageBox.warning(
                self.mw,
                "Segment Everything",
                "Another inference run is still in progress. Wait for it to "
                "finish and try again.",
            )
            return
        except Exception:
            logger.exception("Segment Everything failed")
            QMessageBox.critical(
                self.mw,
                "Segment Everything",
                "Segmentation failed. See the log for details.",
            )
            return
        finally:
            # Read the flag BEFORE closing. ``QProgressDialog.closeEvent``
            # emits ``canceled()``, which Qt wires to the ``cancel()`` slot --
            # so ``close()`` sets ``wasCanceled()`` all by itself. Checking it
            # afterwards reported "cancelled" on every single run and threw
            # away every proposal SAM had just spent seconds producing, with
            # one log line to show for it. After close() the flag it
            # is True; after hide() or reset() it is False. (Verified on Qt
            # 6.11.1 / PyQt6 6.11.0, the versions actually in this venv.)
            cancelled = progress.wasCanceled()
            progress.close()

        if cancelled:
            logger.info("Segment Everything cancelled by the user")
            return
        if not proposals:
            QMessageBox.information(
                self.mw, "Segment Everything", "SAM produced no masks here."
            )
            return

        self.stage_proposals(proposals)

    def stage_proposals(self, proposals):
        """Filter raw proposals and put the survivors into the review overlay.

        Split from :meth:`run` so the filtering and staging can be tested
        without a model.
        """
        pixmap = self.mw.image_label.original_pixmap
        width = pixmap.width() if pixmap is not None else 0
        height = pixmap.height() if pixmap is not None else 0

        existing = [
            annotation["segmentation"]
            for annotations in self.mw.image_label.annotations.values()
            for annotation in annotations
            if annotation.get("segmentation")
        ]

        kept, dropped = mask_filters.filter_mask_proposals(
            proposals,
            width,
            height,
            existing_segmentations=existing,
            min_area=self.min_area,
            max_area_fraction=self.max_area_fraction,
            max_candidates=self.max_candidates,
            overlap_iou=self.overlap_iou,
        )

        if not kept:
            QMessageBox.information(
                self.mw,
                "Segment Everything",
                "Every proposal was filtered out "
                f"({mask_filters.describe_dropped(dropped)}). "
                "Loosen the thresholds and try again.",
            )
            return

        proposals_for_review = [
            {
                "segmentation": proposal["segmentation"],
                "category_name": TEMP_AUTO_CLASS,
                "score": proposal.get("score", 0.0),
                "source": SOURCE,
                "assigned_class": None,
                "temp": True,
            }
            for proposal in kept
        ]
        self.mw.image_label.temp_annotations = proposals_for_review
        # Also park them under this image's key. `temp_annotations` is a single
        # field, not per-image, and `_refresh_dino_temp_for_current` clears it
        # on every image/slice switch (CLAUDE.md) -- so without this, a stray
        # click in the image list would silently discard a batch the user may
        # have spent minutes assigning classes to. Reusing dino_batch_results
        # means the existing re-sync restores them on the way back.
        image_name = self.mw.current_slice or self.mw.image_file_name
        if image_name:
            self.mw.dino_batch_results[image_name] = proposals_for_review
        # A colour for the unassigned state, so the overlay has something to
        # draw against before any class is picked.
        self.mw.image_label.class_colors.setdefault(
            TEMP_AUTO_CLASS,
            self.mw.image_label.class_colors.get(TEMP_AUTO_CLASS)
            or _temp_colour(self.mw),
        )
        self.mw.image_label.update()
        self.mw.image_label.setFocus()

        summary = mask_filters.describe_dropped(dropped)
        message = (
            f"{len(kept)} proposal(s). Click one to assign the active class "
            "(digits 1-9 switch class), then Enter to commit or Escape to discard."
        )
        if summary:
            message += f"  Filtered out: {summary}."
        self.mw.lbl_dino_status.setText(message)
        logger.info(
            "Segment Everything staged %d proposal(s); dropped %s",
            len(kept),
            dropped,
        )


def _temp_colour(main_window):
    from PyQt6.QtGui import QColor

    return QColor(default_class_color(len(main_window.image_label.class_colors)))
