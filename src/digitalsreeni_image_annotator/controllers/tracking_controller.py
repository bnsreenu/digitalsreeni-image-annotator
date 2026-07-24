"""SAM 3 video object-tracking coordination controller (issue #51, ADR-040).

Owns the "track a selected object across a video's frames" workflow. SAM 3's
video predictor propagates one seeded object; this controller turns the seed
(the single selected segmentation on the current frame) into a run, then routes
each per-frame result:

- confident frames (``score >= threshold``) are committed straight to the
  project as ``source == "sam3-track"`` annotations tagged with a shared
  ``track_run`` id (so the whole run can be rolled back in one shot);
- uncertain frames (``0 < score < threshold``) become **temp** results in
  ``dino_batch_results`` (source ``"sam3"``), so the EXISTING DINO batch-review
  pipeline + Enter/Escape event filter handle them verbatim — nothing new to
  wire for review;
- absent frames (``None``) produce nothing.

Everything the model touches is isolated in ``SAM3Utils.track`` /
``_track_blocking`` (the monkeypatch seam); this controller is pure,
stub-testable orchestration.

Undo granularity (ADR-026):

- **Per-frame Ctrl+Z** undoes one frame's tracked annotation — every commit
  calls ``record_history(frame_name)`` before writing, so each frame's own undo
  stack has an entry.
- **Undo Last Track** (:meth:`undo_last_track`) is the bulk convenience: it
  removes every annotation carrying the last run's ``track_run`` id across all
  its frames in one action (also per-frame ``record_history`` first, so it too
  is individually undoable).
"""

import uuid

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
)

from ..core.video_handler import frame_key, parse_frame_index
from ..inference.sam_utils import InferenceBusyError

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class TrackingController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        # The most recent tracking run, for the "Undo Last Track" affordance.
        # {"run_id": str, "frames": [frame_name, ...]} or None.
        self._last_run = None

    # --- gating -------------------------------------------------------------

    def _selected_segmentation(self):
        """The single selected annotation iff it carries a segmentation, else
        ``None``.

        Uses the ADR-023 single-selection resolution
        (``image_label._single_selected_shape``) and additionally requires a
        ``"segmentation"`` key — a pose instance has a bbox but NO segmentation
        (ADR-029), so gating on segmentation excludes it (you can't propagate a
        keypoint skeleton as a mask).
        """
        sel = self.mw.image_label._single_selected_shape()
        if sel is None:
            return None
        if not sel.get("segmentation"):
            return None
        return sel

    def can_track(self):
        """True when object tracking is possible: the active image is a loaded
        video, SAM 3 is loaded, and exactly one segmentation annotation is
        selected. Drives the Tools-menu action's enabled state and guards
        :meth:`run_tracking`."""
        if self.mw.image_controller.current_video() is None:
            return False
        if not self.mw.sam3_utils.loaded:
            return False
        return self._selected_segmentation() is not None

    # --- run ----------------------------------------------------------------

    def run_tracking(self):
        """Propagate the selected object across the video's frames (#51)."""
        if not self.can_track():
            QMessageBox.warning(
                self.mw,
                "Cannot Track",
                "Object tracking needs a loaded video, the SAM 3 model loaded, "
                "and exactly one mask (segmentation) annotation selected on the "
                "current frame.",
            )
            return

        ann = self._selected_segmentation()
        video = self.mw.image_controller.current_video()
        base_name, handler, _info = video

        seed_idx = parse_frame_index(self.mw.current_slice or "")
        if seed_idx is None:
            QMessageBox.warning(
                self.mw, "Cannot Track",
                "Could not determine the current video frame to seed from.",
            )
            return

        class_name = ann["category_name"]
        seg = ann["segmentation"]
        # Ultralytics `bboxes=` wants xyxy pixel coords (same as apply_sam_*).
        xs, ys = seg[0::2], seg[1::2]
        seed_bbox = [min(xs), min(ys), max(xs), max(ys)]

        threshold = self._prompt_tracking_options()
        if threshold is None:
            return  # user cancelled

        # Modal progress dialog: WindowModal blocks GUI-side frame navigation
        # for the duration of the track (the guard against switching frames
        # mid-propagation), while its Cancel button feeds `should_cancel`.
        progress = self._make_progress_dialog()
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        # should_cancel reads QProgressDialog.wasCanceled() from the worker
        # thread — the one tolerated cross-thread touch (a bool getter read
        # under the GIL; the propagation is otherwise Qt-free on the worker).
        def should_cancel():
            return progress.wasCanceled()

        try:
            results = self.mw.sam3_utils.track(
                handler.path, seed_idx, seed_bbox, should_cancel=should_cancel
            )
        except InferenceBusyError:
            progress.close()
            logger.warning("track: another inference is running; skipping")
            QMessageBox.information(
                self.mw, "Busy",
                "Another inference is still running. Wait for it to finish, "
                "then try tracking again.",
            )
            return
        except Exception as e:
            progress.close()
            logger.exception("Object tracking failed")
            QMessageBox.critical(self.mw, "Tracking Error", str(e))
            return
        finally:
            progress.close()

        self._apply_results(results, base_name, seed_idx, class_name, threshold)

    def _apply_results(self, results, base_name, seed_idx, class_name, threshold):
        """Route each ``(frame_idx, result)`` to commit / review / discard."""
        run_id = str(uuid.uuid4())
        committed_frames = []
        uncertain = 0

        for frame_idx, result in results:
            if result is None:
                continue
            # The seed frame already carries the source annotation; committing a
            # tracked copy there would duplicate it.
            if frame_idx == seed_idx:
                continue
            polygon = result.get("segmentation")
            if not polygon:
                continue
            score = float(result.get("score", 0.0))
            frame_name = frame_key(base_name, frame_idx)

            if score >= threshold:
                # Only record the frame as committed if the write actually
                # happened (unknown-class commits no-op) so _last_run["frames"]
                # can't list frames that hold no tracked annotation.
                if self._commit_tracked_result(
                    frame_name, polygon, score, class_name, run_id
                ):
                    committed_frames.append(frame_name)
            elif score > 0:
                # Uncertain → temp-shaped result in dino_batch_results, source
                # "sam3" so the existing review pipeline + event filter handle
                # it. NOT attached to the single `temp_annotations` field —
                # `_refresh_dino_temp_for_current` surfaces it on navigation.
                self.mw.dino_batch_results[frame_name] = [{
                    "segmentation": polygon,
                    "category_name": class_name,
                    "score": score,
                    "source": "sam3",
                    "temp": True,
                }]
                uncertain += 1

        self._last_run = {"run_id": run_id, "frames": committed_frames}

        # Refresh UI + persist ONCE (not per frame).
        self.mw.update_annotation_list()
        self.mw.update_slice_list_colors()
        self.mw.update_video_timeline()
        self.mw.auto_save()

        logger.info(
            "track run %s: %d committed, %d uncertain",
            run_id, len(committed_frames), uncertain,
        )

        if uncertain:
            reply = QMessageBox.question(
                self.mw, "Review Uncertain Frames",
                f"Tracking finished. {uncertain} frame(s) had low-confidence "
                "results. Review them now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.mw.dino_controller._show_dino_batch_review()

    def _commit_tracked_result(self, frame_name, polygon, score, class_name, run_id):
        """Commit one tracked mask to a frame — MIRRORS
        ``DINOController._commit_dino_results``.

        ``record_history(frame_name)`` FIRST (ADR-026: per-frame Ctrl+Z undo),
        then write to ``image_label.annotations`` if the frame is on screen else
        directly to ``all_annotations[frame_name]``. The committed dict carries
        ``source == "sam3-track"`` + the shared ``track_run`` id so
        :meth:`undo_last_track` can find and remove the whole run.
        """
        if class_name not in self.mw.class_mapping:
            logger.warning("track: unknown class '%s'; skipping commit", class_name)
            return False

        current_image = self.mw.current_slice or self.mw.image_file_name
        is_current = frame_name == current_image

        # Snapshot under this frame's OWN key (differs from the on-screen frame
        # for the off-screen commits) so each frame is individually undoable.
        self.mw.annotation_controller.record_history(frame_name)

        if is_current:
            target = self.mw.image_label.annotations
        else:
            if frame_name not in self.mw.all_annotations:
                self.mw.all_annotations[frame_name] = {}
            target = self.mw.all_annotations[frame_name]

        existing = target.get(class_name, [])
        number = max((a.get("number", 0) for a in existing), default=0) + 1
        # Same shape as DINOController._commit_dino_results, plus the track_run
        # tag that lets undo_last_track find the whole run.
        ann = {
            "segmentation": polygon,
            "category_id": self.mw.class_mapping[class_name],
            "category_name": class_name,
            "score": score,
            "source": "sam3-track",
            "track_run": run_id,
            "number": number,
        }
        target.setdefault(class_name, []).append(ann)

        if is_current:
            self.mw.save_current_annotations()
            self.mw.image_label.update()
        return True

    # --- undo the last run --------------------------------------------------

    def undo_last_track(self):
        """Remove every annotation from the last tracking run in one action.

        Finds annotations by exact ``track_run == run_id`` match (per-frame,
        exact-key). ``record_history(frame_name)`` before each frame's write, so
        this bulk removal is itself per-frame undoable. Pre-existing (non-run)
        annotations on those frames are left intact.
        """
        if not self._last_run or not self._last_run.get("frames"):
            QMessageBox.information(
                self.mw, "Undo Last Track", "There is no tracking run to undo."
            )
            return

        run_id = self._last_run["run_id"]
        current_image = self.mw.current_slice or self.mw.image_file_name

        for frame_name in self._last_run["frames"]:
            is_current = frame_name == current_image
            target = (
                self.mw.image_label.annotations
                if is_current
                else self.mw.all_annotations.get(frame_name)
            )
            if not target:
                continue

            self.mw.annotation_controller.record_history(frame_name)
            for cls in list(target.keys()):
                target[cls] = [
                    a for a in target[cls] if a.get("track_run") != run_id
                ]
                if not target[cls]:
                    del target[cls]

            if is_current:
                self.mw.save_current_annotations()
            elif not target:
                # Leave no empty per-frame dict behind (mirrors
                # save_current_annotations' delete-if-empty for the current one).
                self.mw.all_annotations.pop(frame_name, None)

        self._last_run = None
        # Drop any selection referencing a just-removed shape so no floating
        # selection overlay is painted (mirrors _restore_snapshot).
        self.mw.image_label.highlighted_annotations.clear()
        self.mw.update_annotation_list()
        self.mw.update_slice_list_colors()
        self.mw.update_video_timeline()
        self.mw.image_label.update()
        self.mw.auto_save()
        logger.info("undo_last_track: removed run %s", run_id)

    # --- dialogs (factored out so tests can stub them) ----------------------

    def _prompt_tracking_options(self):
        """Confirm dialog with a confidence-threshold spinbox (default 0.5).

        Returns the chosen threshold (float in [0, 1]) or ``None`` if the user
        cancelled. Factored out so tests replace the modal wholesale.
        """
        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Track Selected Object")
        form = QFormLayout(dialog)
        form.addRow(QLabel(
            "Propagate the selected object across the video's frames using "
            "SAM 3.\nFrames scoring below the threshold are queued for review."
        ))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(0.5)
        form.addRow("Confidence threshold:", spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return float(spin.value())
        return None

    def _make_progress_dialog(self):
        """The busy/cancel progress dialog for a track (indeterminate: the whole
        propagation is one blocking ``_run_sync`` call). Factored out so tests
        can substitute a non-modal fake."""
        # Range (0, 0) → an indeterminate "busy" bar; the worker runs the whole
        # clip in one call, so there is no per-frame progress to report here.
        return QProgressDialog(
            "Tracking object across frames…", "Cancel", 0, 0, self.mw
        )
