"""SAM (Segment Anything) coordination controller.

Extracted from `ImageAnnotator`. Owns the SAM tool lifecycle (box,
points), the debounce timer state machine, ADR-013's in-flight
re-entrancy guard, and the model picker dropdown plumbing.

State (`sam_utils`, `sam_inference_timer`, `_sam_inference_in_flight`,
`current_sam_model`) stays on the main window in this phase for the
same reason ProjectController / ImageController state stays there:
external callers (image_label.py, clear_all, the sidebar button
enabling logic) read these attributes directly via `main_window.X`. A
future phase may migrate ownership.

ADR-013 invariants preserved verbatim:
- `_sam_inference_in_flight` flag set BEFORE calling
  `sam_utils.apply_sam_*`, cleared in `finally`.
- `InferenceBusyError` (raised by `sam_utils._run_sync` when the worker
  thread is already running) is swallowed silently — the next user
  click restarts the debounce.
- `change_sam_model` blocks via `_run_sync` event-loop pump; UI stays
  responsive.
"""

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import QMessageBox

from ..inference.sam_utils import InferenceBusyError

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class SAMController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def deactivate_sam_tools(self):
        """Turn off SAM box / points and clear any pending SAM state.

        Called before YOLO predictions overlay their own temp results
        and when the SAM model is unset, so a stale bbox / point set /
        temp prediction can't linger into the next workflow."""
        self.mw.sam_inference_timer.stop()
        self.mw.sam_box_button.setChecked(False)
        self.mw.sam_points_button.setChecked(False)

        image_label = self.mw.image_label
        if image_label.current_tool in ("sam_box", "sam_points"):
            image_label.current_tool = None
        image_label.sam_box_active = False
        image_label.sam_points_active = False
        image_label.sam_bbox = None
        image_label.drawing_sam_bbox = False
        image_label.sam_positive_points = []
        image_label.sam_negative_points = []
        image_label.temp_sam_prediction = None
        image_label.setCursor(Qt.CursorShape.ArrowCursor)

        self.mw.update_ui_for_current_tool()

    def schedule_sam_prediction(self):
        """Restart the debounce timer; inference fires 1s after last click."""
        self.mw.sam_inference_timer.stop()
        self.mw.sam_inference_timer.start(1000)

    def cancel_sam_debounce(self):
        """Stop the SAM debounce timer so a queued inference doesn't
        fire. Does NOT abort an in-flight inference; that case is
        handled by the _sam_inference_in_flight guard (ADR-013).
        Triggered by Escape in ImageLabel while sam_points is active."""
        self.mw.sam_inference_timer.stop()

    def apply_sam_prediction(self):
        # Re-entry guard (ADR-013): the event-loop pump inside _run_sync
        # can deliver this timer fire before the first call returns.
        # Bail and rely on the user clicking again (which restarts the
        # debounce) to issue a fresh inference with the up-to-date
        # point set.
        if self.mw._sam_inference_in_flight:
            return
        self.mw._sam_inference_in_flight = True
        try:
            try:
                if self.mw.image_label.current_tool == "sam_box":
                    if self.mw.image_label.sam_bbox is None:
                        logger.debug("SAM bbox is None")
                        return
                    x1, y1, x2, y2 = self.mw.image_label.sam_bbox
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    prediction = self.mw.sam_utils.apply_sam_prediction(
                        self.mw.current_image, bbox
                    )
                    self.mw.image_label.sam_bbox = None
                elif self.mw.image_label.current_tool == "sam_points":
                    pos_points = self.mw.image_label.sam_positive_points
                    neg_points = self.mw.image_label.sam_negative_points
                    logger.debug(
                        f"Predicting with {len(pos_points)} positive points: {pos_points} "
                        f"and {len(neg_points)} negative points: {neg_points}"
                    )
                    if not pos_points:
                        logger.debug("No positive points for SAM-points")
                        return
                    prediction = self.mw.sam_utils.apply_sam_points(
                        self.mw.current_image,
                        pos_points,
                        neg_points,
                    )
                else:
                    return
            except InferenceBusyError:
                # Re-entry safety net from sam_utils itself. The
                # call-site flag above should catch this first, but if
                # a different caller drives inference concurrently we
                # skip — the user keeps interacting; their next click
                # will restart the debounce.
                return
            except Exception as exc:
                logger.exception("SAM inference failed")
                QMessageBox.critical(
                    self.mw,
                    "SAM Error",
                    f"SAM inference failed:\n\n{exc}\n\n"
                    "See the log for details.",
                )
                return

            if prediction:
                temp_annotation = {
                    "segmentation": prediction["segmentation"],
                    "category_id": self.mw.class_mapping[self.mw.current_class],
                    "category_name": self.mw.current_class,
                    "score": prediction["score"],
                }
                self.mw.image_label.temp_sam_prediction = temp_annotation
                self.mw.image_label.update()
            elif prediction is None:
                QMessageBox.information(
                    self.mw,
                    "SAM",
                    "No mask matches the given constraints. "
                    "Try adjusting the box or point positions."
                )
            else:
                logger.warning("Failed to generate prediction")

            if self.mw.image_label.current_tool == "sam_box":
                self.mw.image_label.sam_bbox = None
                self.mw.image_label.update()
        finally:
            self.mw._sam_inference_in_flight = False

    def accept_sam_prediction(self):
        if self.mw.image_label.temp_sam_prediction:
            self.mw.annotation_controller.record_history()
            new_annotation = self.mw.image_label.temp_sam_prediction
            self.mw.image_label.annotations.setdefault(
                new_annotation["category_name"], []
            ).append(new_annotation)
            self.mw.add_annotation_to_list(new_annotation)
            self.mw.save_current_annotations()
            self.mw.update_slice_list_colors()
            self.mw.image_label.temp_sam_prediction = None
            self.mw.image_label.sam_positive_points = []
            self.mw.image_label.sam_negative_points = []
            self.mw.image_label.update()
            logger.info("SAM prediction accepted, points cleared, and added to annotations.")

    def toggle_sam_box(self):
        # Route through the unified tool choke-point so SAM-box is mutually
        # exclusive with the manual tools and with SAM-points (F).
        if self.mw.sam_box_button.isChecked():
            if self._pose_class_blocks_tool(self.mw.sam_box_button):
                return
            self.mw.activate_tool("sam_box")
        else:
            self.mw.activate_tool(None)

    def toggle_sam_points(self):
        if self.mw.sam_points_button.isChecked():
            if self._pose_class_blocks_tool(self.mw.sam_points_button):
                return
            self.mw.activate_tool("sam_points")
        else:
            self.mw.activate_tool(None)

    def _pose_class_blocks_tool(self, button):
        """A pose class admits only the keypoint tool; refuse SAM on it (#44).

        Unchecks the button and returns True when blocked, mirroring the
        manual-tool guard in ImageAnnotator.toggle_tool.
        """
        if self.mw.current_class in self.mw.keypoint_schemas:
            QMessageBox.warning(
                self.mw,
                "Pose Class",
                f"'{self.mw.current_class}' is a pose class — only the "
                "Keypoint tool can annotate it.",
            )
            button.setChecked(False)
            return True
        return False

    def change_sam_model(self, model_name):
        try:
            self.mw.sam_utils.change_sam_model(model_name)
        except Exception as e:
            from ..core.torch_utils import _is_oom
            logger.exception("Failed to load SAM model '%s'", model_name)
            if _is_oom(e):
                QMessageBox.critical(
                    self.mw,
                    "Not Enough Memory",
                    f"Not enough GPU/system memory to load '{model_name}'.\n\n"
                    "Close other applications or pick a smaller model "
                    "(SAM 2 tiny or small are recommended)."
                )
            else:
                QMessageBox.critical(
                    self.mw,
                    "SAM Model Error",
                    f"Failed to load SAM model '{model_name}':\n\n{str(e)}\n\n"
                    "Check that the model weights are downloadable and that torch "
                    "is correctly installed for your platform / GPU."
                )
            self.mw.sam_model_selector.setCurrentIndex(0)
            return

        self.mw.current_sam_model = self.mw.sam_utils.current_sam_model

        if model_name != "Pick a SAM Model":
            logger.info(f"Changed SAM model to: {model_name}")
            # One-time dialog if CUDA exists but the torch wheels can't
            # run it (e.g. Pascal sm_61 on torch>=2.8) — upstream #57.
            from ..core.torch_utils import maybe_warn_cpu_fallback
            maybe_warn_cpu_fallback(self.mw)
        else:
            self.deactivate_sam_tools()
            logger.info("SAM model unset")
