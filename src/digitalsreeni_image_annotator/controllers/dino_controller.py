"""DINO (LLM-assisted detection) coordination controller.

Extracted from `ImageAnnotator`. Owns:

- DINO model picker plumbing (preset / custom-path resolution,
  on-demand HuggingFace Hub download)
- Single-image and batch detection workflows (DINO produces bboxes →
  SAM refines to masks)
- Temp-annotation review state: accept / reject pending DINO results,
  navigate batch review across mixed regular-images + multi-dim slices
- The application-wide `DINOReviewEventFilter` that lets Enter /
  Escape accept-or-reject pending DINO masks regardless of which
  widget has focus

State (`dino_utils`, `dino_model_loaded`, `dino_custom_model_path`,
`dino_batch_results`) stays on the main window in this phase — same
deferral as prior controllers. Widgets that own DINO configuration
(`dino_phrase_panel`, `dino_class_table`, `dino_model_selector`,
`dino_batch_mode`, `lbl_dino_status`, `btn_detect_*`, `dino_browse_row`,
`lbl_dino_custom`) also stay on the main window.

The temp-annotation review machinery (Temp-* class handling) lives
here too — it was originally a separate workflow for YOLO predictions
but is now shared with DINO and most easily co-located.
"""

import os
import traceback

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QTextEdit,
)

from ..core.constants import default_class_color
from ..core.keypoint_schema import schema_k


class DINOReviewEventFilter(QObject):
    """Application-wide event filter that lets Enter / Escape accept or
    reject pending DINO temp_annotations regardless of which widget has
    focus. Without this, clicking a slice/image entry in a list moves
    focus there and Enter is consumed by the list's itemActivated
    handler before it can reach ImageLabel.keyPressEvent.

    Suppressed when a modal dialog is active or focus is on a text-input
    widget so we don't break dialog default-button behaviour or
    in-cell editing.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = event.key()
        if key not in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return False
        app = QApplication.instance()
        if app is None or app.activeModalWidget() is not None:
            return False
        focused = app.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit)):
            return False
        temp = self.main_window.image_label.temp_annotations
        if not temp or not any(a.get("source") == "dino" for a in temp):
            return False
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.main_window.accept_dino_results()
        else:
            self.main_window.reject_dino_results()
        return True


class DINOController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    # --- Model picker plumbing ---

    def _resolve_dino_model_path(self, model_name):
        """Return the canonical local path for a preset DINO model, or None if unknown."""
        from ..inference.dino_utils import GDINO_MODEL_PATHS
        return GDINO_MODEL_PATHS.get(model_name)

    def _on_dino_model_changed(self, text):
        """Selection → ready state. Downloads happen lazily on first Detect."""
        self.mw.dino_browse_row.setVisible(text == "Custom / fine-tuned (browse)")

        if text == "Pick a DINO Model":
            self.mw.dino_model_loaded = False
            self.mw.lbl_dino_status.setText("No DINO model loaded")
            self.mw.btn_detect_single.setEnabled(False)
            self.mw.btn_detect_batch.setEnabled(False)
            return

        if text == "Custom / fine-tuned (browse)":
            if (
                self.mw.dino_custom_model_path
                and os.path.exists(self.mw.dino_custom_model_path)
            ):
                self.mw.dino_model_loaded = True
                self.mw.lbl_dino_status.setText(
                    f"Ready: {os.path.basename(self.mw.dino_custom_model_path)}"
                )
                self.mw.btn_detect_single.setEnabled(True)
                self.mw.btn_detect_batch.setEnabled(True)
            else:
                self.mw.dino_model_loaded = False
                self.mw.lbl_dino_status.setText("Browse for a custom model folder")
                self.mw.btn_detect_single.setEnabled(False)
                self.mw.btn_detect_batch.setEnabled(False)
            return

        self.mw.dino_model_loaded = True
        self.mw.btn_detect_single.setEnabled(True)
        self.mw.btn_detect_batch.setEnabled(True)
        model_path = self._resolve_dino_model_path(text)
        if model_path and os.path.exists(model_path):
            self.mw.lbl_dino_status.setText(f"Ready: {text}")
        else:
            self.mw.lbl_dino_status.setText(f"{text} — will download on first detection")

    def _ensure_dino_model_downloaded(self, model_name):
        """If the preset model isn't on disk yet, download it. Returns success."""
        if model_name in ("Pick a DINO Model", "Custom / fine-tuned (browse)"):
            return True
        model_path = self._resolve_dino_model_path(model_name)
        if model_path and os.path.exists(model_path):
            return True

        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self.mw, "Missing Dependency",
                f"Cannot download {model_name}: the huggingface_hub package "
                "is not installed.\n\nRun:\n    pip install huggingface_hub",
            )
            return False

        self.mw.lbl_dino_status.setText(f"Downloading {model_name}...")
        QApplication.processEvents()
        try:
            downloaded = self.mw.dino_utils.download_model(model_name)
        except Exception as e:
            QMessageBox.critical(self.mw, "Download Failed", f"{model_name}:\n{e}")
            return False
        if not downloaded:
            QMessageBox.critical(
                self.mw, "Download Failed",
                f"Could not download {model_name} from Hugging Face Hub.",
            )
            return False
        return True

    def browse_dino_model(self):
        path = QFileDialog.getExistingDirectory(self.mw, "Select DINO Model Folder")
        if path:
            self.mw.dino_custom_model_path = path
            self.mw.lbl_dino_custom.setText(os.path.basename(path))
            self._on_dino_model_changed(self.mw.dino_model_selector.currentText())

    def on_dino_class_row_changed(self):
        name = self.mw.dino_class_table.selected_class_name()
        self.mw.dino_phrase_panel.set_active_class(name)

    def _build_dino_class_configs(self):
        """Build class_configs from threshold table + phrase panel."""
        configs = []
        for cfg in self.mw.dino_class_table.get_class_configs():
            phrases = self.mw.dino_phrase_panel.get_phrases_for(cfg["name"])
            configs.append({
                "name": cfg["name"],
                "phrases": phrases,
                "box_thr": cfg["box_thr"],
                "txt_thr": cfg["txt_thr"],
                "nms_thr": cfg["nms_thr"],
            })
        return configs

    # --- Detection workflows ---

    def run_dino_detection_single(self):
        if not self.mw.dino_model_loaded:
            QMessageBox.warning(self.mw, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.mw.sam_utils.current_sam_model:
            QMessageBox.warning(
                self.mw, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.mw.current_image or self.mw.current_image.isNull():
            QMessageBox.warning(self.mw, "No Image",
                                "Please load an image first.")
            return

        model_name = self.mw.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self.mw, "No Classes",
                                "Please add at least one class with phrases.")
            return

        from ..core.torch_utils import maybe_warn_cpu_fallback
        maybe_warn_cpu_fallback(self.mw)

        self.mw.btn_detect_single.setEnabled(False)
        self.mw.btn_detect_batch.setEnabled(False)

        # Clear any stale temp annotations before starting detection so an
        # accept from a previous run doesn't bleed into the results handler.
        self.mw.image_label.temp_annotations = []

        if not self._ensure_dino_model_downloaded(model_name):
            self.mw.btn_detect_single.setEnabled(True)
            self.mw.btn_detect_batch.setEnabled(True)
            return

        self.mw.lbl_dino_status.setText("Detecting...")
        QApplication.processEvents()

        print(f"[DINO] detect_single: model={model_name!r} class_configs={class_configs}")
        try:
            results = self.mw.dino_utils.detect(
                self.mw.current_image, class_configs,
                model_name=model_name,
                custom_model_path=self.mw.dino_custom_model_path,
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.mw, "DINO Error", str(e))
            self.mw.btn_detect_single.setEnabled(True)
            self.mw.btn_detect_batch.setEnabled(True)
            self.mw.lbl_dino_status.setText("Detection failed.")
            return

        self.mw.btn_detect_single.setEnabled(True)
        self.mw.btn_detect_batch.setEnabled(True)

        if results is None:
            print("[DINO] detect_single: results=None (model resolution failure)")
            self.mw.lbl_dino_status.setText("No detections.")
            return

        print(f"[DINO] detect_single: got {len(results)} result(s)")
        if results:
            for i, r in enumerate(results[:3]):
                print(f"[DINO]   result[{i}] class={r['class_name']!r} score={r['score']:.3f} bbox={r['bbox']}")

        if not results:
            self.mw.lbl_dino_status.setText("No detections found.")
            return

        self.mw.lbl_dino_status.setText(f"{len(results)} detection(s). Running SAM...")
        QApplication.processEvents()

        bboxes = [r["bbox"] for r in results]
        print(f"[SAM] batch call: {len(bboxes)} bbox(es), first 3 = {bboxes[:3]}")
        try:
            sam_results = self.mw.sam_utils.apply_sam_predictions_batch(
                self.mw.current_image, bboxes
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self.mw, "SAM Error", str(e))
            self.mw.lbl_dino_status.setText("SAM segmentation failed.")
            return

        if sam_results is None:
            print("[SAM] batch returned None (no SAM model loaded)")
            QMessageBox.warning(self.mw, "SAM Error",
                                "Failed to segment detections with SAM.")
            self.mw.lbl_dino_status.setText("SAM segmentation failed.")
            return

        n_errors = sum(1 for s in sam_results if "error" in s)
        n_ok = sum(1 for s in sam_results if "segmentation" in s)
        print(f"[SAM] batch returned {len(sam_results)} result(s): {n_ok} ok, {n_errors} error(s)")

        # Honor the batch-mode dropdown for the single-image case too:
        # "Auto-accept" means commit straight to annotations without
        # showing the temp-review overlay. The dropdown name is "batch"
        # historically but it controls both paths.
        image_name = self.mw.current_slice or self.mw.image_file_name
        auto_accept = (
            self.mw.dino_batch_mode.currentText() == "Auto-accept all detections"
        )
        if auto_accept:
            print(f"[DINO] detect_single: auto_accept=True, committing {len(results)} result(s)")
            try:
                self._commit_dino_results(image_name, results, sam_results)
            except Exception as e:
                print(f"[DINO] _commit_dino_results failed: {e}")
                traceback.print_exc()
            n_committed = sum(1 for s in sam_results if "error" not in s)
            self.mw.image_label.temp_annotations = []
            self.mw.image_label.update()
            self.mw.update_annotation_list()
            # Refresh slice list so the freshly-annotated slice picks
            # up the highlight color; review-mode's accept_dino_results
            # already does this, the auto-accept path didn't.
            self.mw.update_slice_list_colors()
            self.mw.auto_save()
            self.mw.lbl_dino_status.setText(
                f"Loaded: {model_name}  |  {n_committed} mask(s) auto-accepted"
            )
            print(f"[DINO] auto-accept: committed {n_committed} mask(s) to {image_name}")
            return

        # Review mode
        temp_annotations = []
        for r, s in zip(results, sam_results):
            if "error" in s:
                print(f"[SAM]   failed for {r['class_name']}: {s['error']}")
                continue
            temp_annotations.append({
                "segmentation": s["segmentation"],
                "category_name": r["class_name"],
                "score": r["score"],
                "source": "dino",
                "temp": True,
            })

        self.mw.image_label.temp_annotations = temp_annotations
        QTimer.singleShot(0, self.mw.image_label.setFocus)
        self.mw.image_label.update()
        self.mw.lbl_dino_status.setText(
            f"Loaded: {model_name}  |  {len(temp_annotations)} mask(s) ready"
        )
        print(f"[DINO] detection complete: {len(results)} boxes, {len(temp_annotations)} masks attached to canvas")

    def run_dino_detection_batch(self):
        if not self.mw.dino_model_loaded:
            QMessageBox.warning(self.mw, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.mw.sam_utils.current_sam_model:
            QMessageBox.warning(
                self.mw, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.mw.all_images:
            QMessageBox.warning(self.mw, "No Images",
                                "Please load images first.")
            return

        model_name = self.mw.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self.mw, "No Classes",
                                "Please add at least one class with phrases.")
            return

        from ..core.torch_utils import maybe_warn_cpu_fallback
        maybe_warn_cpu_fallback(self.mw)

        # Prevent stale temp annotations from a prior single-image review from
        # confusing the batch results handler or the DINOReviewEventFilter.
        self.mw.image_label.temp_annotations = []

        if not self._ensure_dino_model_downloaded(model_name):
            return

        auto_accept = (
            self.mw.dino_batch_mode.currentText() == "Auto-accept all detections"
        )
        print(f"[DINO] detect_batch: auto_accept={auto_accept}")

        # Build a flat list of (display_name, qimage) work items covering
        # both regular images (loaded from disk) and multi-dim image
        # slices (already QImages in memory). Slices live in
        # self.mw.image_slices[base_name], indexed by their slice_name
        # (e.g. "stack_T1_Z1_C1"). The earlier implementation only
        # iterated self.all_images and skipped multi-slice entries with
        # a console warning, leaving slice-based projects unable to use
        # Detect All.
        work_items = self._collect_dino_batch_work_items()
        if not work_items:
            QMessageBox.information(
                self.mw, "Detect All Images",
                "No images or slices available to process."
            )
            return
        total = len(work_items)

        progress = QProgressDialog("Running LLM Detection...", "Cancel", 0, total, self.mw)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        for idx, (image_name, qimage) in enumerate(work_items):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            QApplication.processEvents()

            try:
                results = self.mw.dino_utils.detect(
                    qimage, class_configs,
                    model_name=model_name,
                    custom_model_path=self.mw.dino_custom_model_path,
                )
            except Exception as e:
                print(f"  DINO failed for {image_name}: {e}")
                continue

            if not results:
                continue

            bboxes = [r["bbox"] for r in results]
            try:
                sam_results = self.mw.sam_utils.apply_sam_predictions_batch(
                    qimage, bboxes
                )
            except Exception as e:
                print(f"  SAM failed for {image_name}: {e}")
                continue
            if sam_results is None:
                continue

            if auto_accept:
                self._commit_dino_results(image_name, results, sam_results)
            else:
                self._store_dino_batch_results(image_name, results, sam_results)

        progress.setValue(total)
        progress.close()

        if auto_accept:
            QMessageBox.information(
                self.mw, "Batch Detection Complete",
                "Detections have been saved to annotations."
            )
            self.mw.update_annotation_list()
            self.mw.update_slice_list_colors()
            self.mw.auto_save()
        else:
            self._show_dino_batch_review()

    def _collect_dino_batch_work_items(self):
        """Return a flat ``[(name, QImage), …]`` list for batch DINO.

        Regular images are loaded from disk via PIL → QImage. Multi-dim
        images contribute one entry per slice from ``self.mw.image_slices``;
        slices that haven't been materialised yet (the parent image was
        never opened in this session) are skipped with a console log.
        """
        from PIL import Image as PILImage
        items = []
        for img_info in self.mw.all_images:
            file_name = img_info["file_name"]
            if img_info.get("is_multi_slice", False):
                base_name = os.path.splitext(file_name)[0]
                slices = self.mw.image_slices.get(base_name, [])
                if not slices:
                    print(f"  Skipping multi-slice image '{file_name}': "
                          "no slices loaded (open the image first to "
                          "materialise its slices).")
                    continue
                for slice_name, qimage in slices:
                    items.append((slice_name, qimage))
            else:
                image_path = self.mw.image_paths.get(file_name)
                if not image_path or not os.path.exists(image_path):
                    print(f"  Skipping '{file_name}': missing image path.")
                    continue
                try:
                    pil_img = PILImage.open(image_path).convert("RGB")
                    qimage = QImage(
                        pil_img.tobytes(),
                        pil_img.width,
                        pil_img.height,
                        pil_img.width * 3,
                        QImage.Format.Format_RGB888,
                    )
                    items.append((file_name, qimage))
                except Exception as e:
                    print(f"  Skipping '{file_name}': failed to load ({e}).")
        print(f"[DINO] batch work items: {len(items)} total")
        return items

    def _commit_dino_results(self, image_name, dino_results, sam_results):
        """Commit DINO+SAM results to annotations for a single image.

        If image_name is the currently-displayed image, route through
        image_label.annotations so the canvas reflects the change and the
        next save_current_annotations() doesn't overwrite the additions.
        Otherwise write directly to the project-level cache.
        """
        current_image = self.mw.current_slice or self.mw.image_file_name
        is_current = image_name == current_image

        # Snapshot under this image's own key (may differ from the on-screen
        # image for batch commits) so the additions are undoable (ADR-026).
        self.mw.annotation_controller.record_history(image_name)

        if is_current:
            target = self.mw.image_label.annotations
        else:
            if image_name not in self.mw.all_annotations:
                self.mw.all_annotations[image_name] = {}
            target = self.mw.all_annotations[image_name]

        for r, s in zip(dino_results, sam_results):
            if "error" in s:
                continue
            class_name = r["class_name"]
            if class_name not in self.mw.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            existing = target.get(class_name, [])
            number = max((a.get("number", 0) for a in existing), default=0) + 1
            ann = {
                "segmentation": s["segmentation"],
                "category_id": self.mw.class_mapping[class_name],
                "category_name": class_name,
                "score": r["score"],
                "source": "dino",
                "number": number,
            }
            target.setdefault(class_name, []).append(ann)

        if is_current:
            self.mw.save_current_annotations()
            self.mw.image_label.update()

    def _store_dino_batch_results(self, image_name, dino_results, sam_results):
        """Store results for batch review mode."""
        valid = []
        for r, s in zip(dino_results, sam_results):
            if "error" not in s:
                valid.append({
                    "segmentation": s["segmentation"],
                    "category_name": r["class_name"],
                    "score": r["score"],
                    "source": "dino",
                    "temp": True,
                })
        self.mw.dino_batch_results[image_name] = valid

    def _show_dino_batch_review(self):
        """Navigate to first image with batch results for review.

        If the next entry refers to an image/slice that's no longer in
        the project (e.g. the source was removed between detection and
        review), pop the orphan and try the next entry so the user
        doesn't get stuck with un-reviewable results.
        """
        if not self.mw.dino_batch_results:
            QMessageBox.information(self.mw, "Batch Detection",
                                    "No detections found in any image.")
            return
        while self.mw.dino_batch_results:
            first = next(iter(self.mw.dino_batch_results))
            if self._navigate_to_image_or_slice(first):
                return
            print(f"[DINO] dropping orphan batch result for {first!r} "
                  "(no matching image or slice in project)")
            self.mw.dino_batch_results.pop(first, None)
        QMessageBox.warning(
            self.mw, "Batch Detection",
            "Detections were produced but none of them map to an image "
            "or slice still in the project. Results discarded.",
        )

    def _navigate_to_image_or_slice(self, name):
        """Switch the UI to a regular image or a slice by name.

        Returns True if a match was found and the switch was issued.
        Used by batch-review navigation, which mixes regular image
        names and slice names in ``dino_batch_results``.
        """
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            if item and item.text() == name:
                self.mw.image_list.setCurrentRow(i)
                self.mw.switch_image(item)
                return True
        for base_name, slices in self.mw.image_slices.items():
            if not any(s_name == name for s_name, _ in slices):
                continue
            for i in range(self.mw.image_list.count()):
                item = self.mw.image_list.item(i)
                if not item:
                    continue
                file_name = item.text()
                if os.path.splitext(file_name)[0] == base_name:
                    self.mw.image_list.setCurrentRow(i)
                    self.mw.switch_image(item)
                    for s_i in range(self.mw.slice_list.count()):
                        s_item = self.mw.slice_list.item(s_i)
                        if s_item and s_item.text() == name:
                            self.mw.slice_list.setCurrentRow(s_i)
                            self.mw.switch_slice(s_item)
                            return True
                    break
            return False
        return False

    def _refresh_dino_temp_for_current(self):
        """Sync ``image_label.temp_annotations`` to whatever the
        currently-displayed image/slice has stored in
        ``dino_batch_results``. Called from switch_slice / switch_image.

        Why this exists: ``temp_annotations`` is a single field on
        ``ImageLabel``, not a per-image cache. Without this sync, masks
        from the previously-viewed image bleed onto every slice the
        user navigates to.
        """
        new_image = self.mw.current_slice or self.mw.image_file_name
        pending = self.mw.dino_batch_results.get(new_image, []) if new_image else []
        if pending:
            self.mw.image_label.temp_annotations = list(pending)
            self.mw.lbl_dino_status.setText(
                f"Review: {new_image}  ({len(pending)} detection(s))"
            )
            QTimer.singleShot(0, self.mw.image_label.setFocus)
        else:
            if self.mw.image_label.temp_annotations:
                print("[DINO] temp annotations cleared on switch "
                      f"(no pending batch results for {new_image!r})")
            self.mw.image_label.temp_annotations = []
        self.mw.image_label.update()

    def accept_dino_results(self):
        """Accept current temp_annotations (called from keyPressEvent)."""
        if not self.mw.image_label.temp_annotations:
            return
        image_name = self.mw.current_slice or self.mw.image_file_name

        self.mw.annotation_controller.record_history(image_name)

        for ann in self.mw.image_label.temp_annotations:
            class_name = ann["category_name"]
            if class_name not in self.mw.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            new_ann = {
                "segmentation": ann["segmentation"],
                "category_id": self.mw.class_mapping[class_name],
                "category_name": class_name,
                "score": ann.get("score", 0.0),
                "source": "dino",
            }
            self.mw.image_label.annotations.setdefault(class_name, []).append(new_ann)
            self.mw.add_annotation_to_list(new_ann)

        self.mw.image_label.temp_annotations = []
        self.mw.dino_batch_results.pop(image_name, None)
        if self.mw.dino_batch_results:
            self._show_dino_batch_review()
        self.mw.save_current_annotations()
        self.mw.update_slice_list_colors()
        self.mw.image_label.update()
        self.mw.lbl_dino_status.setText("Results accepted.")
        print("DINO results accepted.")

    def reject_dino_results(self):
        """Discard current temp_annotations."""
        self.mw.image_label.temp_annotations = []
        image_name = self.mw.current_slice or self.mw.image_file_name
        self.mw.dino_batch_results.pop(image_name, None)
        if self.mw.dino_batch_results:
            self._show_dino_batch_review()
        self.mw.image_label.update()
        self.mw.lbl_dino_status.setText("Results discarded.")
        print("DINO results discarded.")

    # --- Temp-class review workflow (shared with YOLO predictions) ---

    def has_visible_temp_classes(self):
        for i in range(self.mw.class_list.count()):
            item = self.mw.class_list.item(i)
            if (
                item.text().startswith("Temp-")
                and item.checkState() == Qt.CheckState.Checked
            ):
                return True
        return False

    def add_temp_classes(self, temp_annotations):
        for temp_class_name, annotations in temp_annotations.items():
            if temp_class_name not in self.mw.image_label.class_colors:
                color = QColor(
                    default_class_color(len(self.mw.image_label.class_colors))
                )
                self.mw.image_label.class_colors[temp_class_name] = color
            self.mw.image_label.annotations[temp_class_name] = annotations

        self.mw.update_class_list()

    def verify_current_class(self):
        if (
            self.mw.current_class is None
            or self.mw.current_class not in self.mw.class_mapping
        ):
            if self.mw.class_list.count() > 0:
                self.mw.class_list.setCurrentRow(0)
                self.mw.on_class_selected(self.mw.class_list.item(0))
            else:
                self.mw.current_class = None
                self.mw.disable_annotation_tools()

    def accept_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.mw.class_list.findItems(
                "Temp-*", Qt.MatchFlag.MatchWildcard
            )
            if item.checkState() == Qt.CheckState.Checked
        ]

        for temp_class_name in visible_temp_classes:
            permanent_class_name = temp_class_name[5:]
            if permanent_class_name not in self.mw.image_label.annotations:
                self.mw.add_class(
                    permanent_class_name,
                    self.mw.image_label.class_colors[temp_class_name],
                )

            if temp_class_name in self.mw.keypoint_schemas:
                temp_schema = self.mw.keypoint_schemas.pop(temp_class_name)
                existing_schema = self.mw.keypoint_schemas.get(permanent_class_name)
                if existing_schema is None:
                    self.mw.keypoint_schemas[permanent_class_name] = temp_schema
                elif schema_k(existing_schema) != schema_k(temp_schema):
                    QMessageBox.warning(
                        self.mw,
                        "Keypoint Schema Mismatch",
                        f"Predicted pose instances for '{permanent_class_name}' have "
                        f"{schema_k(temp_schema)} keypoints, but the class's existing "
                        f"schema has {schema_k(existing_schema)}. Keeping the existing "
                        "schema; the newly accepted instances may not render or edit "
                        "correctly until you redefine it."
                    )
                # else: existing schema already matches (likely has better hand-authored
                # names than the generic kp0..kpK-1 placeholder) -- keep it, drop the temp one.

            current_max = max(
                [
                    ann.get("number", 0)
                    for ann in self.mw.image_label.annotations.get(
                        permanent_class_name, []
                    )
                ]
                + [0]
            )

            for annotation in self.mw.image_label.annotations[temp_class_name]:
                current_max += 1
                annotation["category_name"] = permanent_class_name
                annotation["number"] = current_max
                self.mw.image_label.annotations.setdefault(
                    permanent_class_name, []
                ).append(annotation)

            del self.mw.image_label.annotations[temp_class_name]
            del self.mw.image_label.class_colors[temp_class_name]

        self.mw.update_class_list()
        current_name = self.mw.current_slice or self.mw.image_file_name
        self.mw.all_annotations[current_name] = self.mw.image_label.annotations
        self.mw.update_annotation_list()
        self.mw.image_label.update()
        self.mw.save_current_annotations()

        self.select_first_primary_class()
        self.verify_current_class()

        QMessageBox.information(
            self.mw,
            "Annotations Accepted",
            "Temporary annotations have been accepted and added to the permanent classes.",
        )

    def select_first_primary_class(self):
        for i in range(self.mw.class_list.count()):
            item = self.mw.class_list.item(i)
            if not item.text().startswith("Temp-"):
                self.mw.class_list.setCurrentItem(item)
                self.mw.on_class_selected(item)
                break

    def reject_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.mw.class_list.findItems(
                "Temp-*", Qt.MatchFlag.MatchWildcard
            )
            if item.checkState() == Qt.CheckState.Checked
        ]

        for temp_class_name in visible_temp_classes:
            if temp_class_name in self.mw.image_label.annotations:
                del self.mw.image_label.annotations[temp_class_name]
            if temp_class_name in self.mw.image_label.class_colors:
                del self.mw.image_label.class_colors[temp_class_name]
            self.mw.keypoint_schemas.pop(temp_class_name, None)

        self.mw.update_class_list()
        self.mw.image_label.update()

    def check_temp_annotations(self):
        temp_classes = [
            class_name
            for class_name in self.mw.image_label.annotations.keys()
            if class_name.startswith("Temp-")
        ]
        if temp_classes:
            reply = QMessageBox.question(
                self.mw,
                "Temporary Annotations",
                "There are temporary annotations that will be discarded. Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                for temp_class in temp_classes:
                    del self.mw.image_label.annotations[temp_class]
                    del self.mw.image_label.class_colors[temp_class]
                self.mw.update_class_list()
                self.mw.update_annotation_list()
                return True
            return False
        return True

    def remove_all_temp_annotations(self):
        for image_name in list(self.mw.all_annotations.keys()):
            for class_name in list(self.mw.all_annotations[image_name].keys()):
                if class_name.startswith("Temp-"):
                    del self.mw.all_annotations[image_name][class_name]
            if not self.mw.all_annotations[image_name]:
                del self.mw.all_annotations[image_name]

        for class_name in list(self.mw.image_label.class_colors.keys()):
            if class_name.startswith("Temp-"):
                del self.mw.image_label.class_colors[class_name]

        self.mw.update_class_list()
        self.mw.update_annotation_list()
        self.mw.image_label.update()
