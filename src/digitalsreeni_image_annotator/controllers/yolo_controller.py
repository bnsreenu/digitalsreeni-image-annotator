"""YOLO training / prediction coordination controller.

Extracted from `ImageAnnotator`. Owns:

- The YOLO menu (Training submenu + Prediction Settings submenu)
- Pre-trained model loading and dataset preparation
- Training: dialog wiring, the `TrainingThread` worker, progress
  callback chain, finish handler
- Prediction: model loading via `LoadPredictionModelDialog`, the
  confidence-threshold dialog, single-image and multi-image prediction
- Result post-processing (`process_yolo_results`) that converts YOLO
  output into temp annotations for the user to review

State (`yolo_trainer`, `training_thread`, `training_dialog`) stays on
the main window — the menu actions and signal connections are
addressed from elsewhere as `main_window.X`, and `training_dialog` is
referenced via `hasattr(self, "training_dialog")` to lazily initialize.
"""

import copy
import os

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..dialogs.yolo_trainer import (
    LoadPredictionModelDialog,
    TrainingInfoDialog,
    YOLOTrainer,
)


def build_yolo_train_opts(epochs, *, cos_lr, lr0, patience):
    """Map the Train-dialog knobs to Ultralytics ``train()`` kwargs (issue #85).

    "Off" must mean a genuinely constant LR to match the SAM schedule toggle:
    ``cos_lr=False`` + ``lrf=1.0`` (no decay) + ``warmup_epochs=0`` (no ramp).
    "On" warms up over the first ~10% of epochs then cosine-decays to a 10%
    floor. Pure so the on/off mapping is unit-testable without the GUI.
    """
    return {
        "cos_lr": cos_lr,
        "lr0": lr0,
        "lrf": 0.1 if cos_lr else 1.0,
        "warmup_epochs": max(1, round(0.1 * epochs)) if cos_lr else 0,
        "patience": patience,
    }


class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, yolo_trainer, epochs, imgsz, train_opts=None):
        super().__init__()
        self.yolo_trainer = yolo_trainer
        self.epochs = epochs
        self.imgsz = imgsz
        self.train_opts = train_opts or {}

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs, imgsz=self.imgsz, **self.train_opts
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))


class YOLOController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        # Latched after the MLflow UI server is started once for the session
        # (mirrors SAMTrainController); a failed launch leaves it False to retry.
        self._mlflow_ui_started = False

    def setup_yolo_menu(self):
        yolo_menu = self.mw.menuBar().addMenu("&YOLO (beta)")

        training_submenu = yolo_menu.addMenu("Training")

        load_pretrained_action = QAction("Load Pre-trained Model", self.mw)
        load_pretrained_action.triggered.connect(self.load_yolo_model)
        training_submenu.addAction(load_pretrained_action)

        prepare_data_action = QAction("Prepare YOLO Dataset", self.mw)
        prepare_data_action.triggered.connect(self.prepare_yolo_dataset)
        training_submenu.addAction(prepare_data_action)

        load_yaml_action = QAction("Load Dataset YAML", self.mw)
        load_yaml_action.triggered.connect(self.load_yolo_yaml)
        training_submenu.addAction(load_yaml_action)

        train_action = QAction("Train Model", self.mw)
        train_action.triggered.connect(self.show_train_dialog)
        training_submenu.addAction(train_action)

        save_model_action = QAction("Save Model", self.mw)
        save_model_action.triggered.connect(self.save_yolo_model)
        training_submenu.addAction(save_model_action)

        prediction_submenu = yolo_menu.addMenu("Prediction Settings")

        load_model_action = QAction("Load Model", self.mw)
        load_model_action.triggered.connect(self.load_prediction_model)
        prediction_submenu.addAction(load_model_action)

        set_threshold_action = QAction("Set Confidence Threshold", self.mw)
        set_threshold_action.triggered.connect(self.set_confidence_threshold)
        prediction_submenu.addAction(set_threshold_action)

    def initialize_yolo_trainer(self):
        if hasattr(self.mw, "current_project_dir"):
            self.mw.yolo_trainer = YOLOTrainer(self.mw.current_project_dir, self.mw)
        else:
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )

    def load_yolo_model(self):
        if not hasattr(self.mw, "current_project_dir"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        if self.mw.yolo_trainer.load_model():
            QMessageBox.information(
                self.mw, "Model Loaded", "YOLO model loaded successfully."
            )
        else:
            QMessageBox.warning(
                self.mw, "Load Cancelled", "Model loading was cancelled."
            )

    def prepare_yolo_dataset(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        # YOLO training needs a non-empty validation set; hold some images out
        # by default (0 keeps everything in train, but val/ will then be empty).
        from .io_controller import prompt_validation_split

        val_split, ok = prompt_validation_split(self.mw)
        if not ok:
            return

        try:
            yaml_path = self.mw.yolo_trainer.prepare_dataset(val_split)
            QMessageBox.information(
                self.mw,
                "Dataset Prepared",
                f"YOLO dataset prepared successfully ({val_split}% validation).\n"
                f"YAML file: {yaml_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self.mw,
                "Error",
                f"An error occurred while preparing the dataset: {str(e)}",
            )

    def load_yolo_yaml(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            if self.mw.yolo_trainer.load_yaml():
                QMessageBox.information(
                    self.mw, "YAML Loaded", "Dataset YAML loaded successfully."
                )
            else:
                QMessageBox.warning(
                    self.mw, "Load Cancelled", "YAML loading was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self.mw,
                "Error",
                f"An error occurred while loading the YAML file: {str(e)}",
            )

    def save_yolo_model(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw, "No Model", "Please train or load a YOLO model first."
            )
            return

        try:
            if self.mw.yolo_trainer.save_model():
                QMessageBox.information(
                    self.mw, "Model Saved", "YOLO model saved successfully."
                )
            else:
                QMessageBox.warning(
                    self.mw, "Save Cancelled", "Model saving was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self.mw, "Error", f"An error occurred while saving the model: {str(e)}"
            )

    def load_prediction_model(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        dialog = LoadPredictionModelDialog(self.mw)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_path = dialog.model_path
            yaml_path = dialog.yaml_path
            if model_path and yaml_path:
                try:
                    result, message = self.mw.yolo_trainer.load_prediction_model(
                        model_path, yaml_path
                    )
                    if result:
                        QMessageBox.information(
                            self.mw,
                            "Model Loaded",
                            "YOLO model and YAML file loaded successfully for prediction.",
                        )
                        if message:
                            QMessageBox.warning(
                                self.mw, "Class Mismatch Warning", message
                            )
                    else:
                        QMessageBox.critical(
                            self.mw,
                            "Error Loading Model",
                            f"Could not load the model or YAML file: {message}",
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self.mw, "Error", f"An error occurred: {str(e)}"
                    )
            else:
                QMessageBox.warning(
                    self.mw,
                    "Files Required",
                    "Both model and YAML files are required for prediction.",
                )

    def show_train_dialog(self):
        if not self.mw.yolo_trainer:
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return
        if not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw, "No Model", "Please load a pre-trained model first."
            )
            return
        if not self.mw.yolo_trainer.yaml_path:
            QMessageBox.warning(
                self.mw, "No Dataset", "Please prepare or load a dataset YAML first."
            )
            return

        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Train YOLO Model")
        layout = QVBoxLayout()

        epochs_label = QLabel("Number of Epochs:")
        epochs_input = QLineEdit("100")
        layout.addWidget(epochs_label)
        layout.addWidget(epochs_input)

        imgsz_label = QLabel("Image Size:")
        imgsz_input = QLineEdit("640")
        layout.addWidget(imgsz_label)
        layout.addWidget(imgsz_input)

        # LR schedule + early stopping (issue bnsreenu#85). The train/val split
        # itself is fixed at "Prepare YOLO Dataset" time (#83) — these only
        # control the optimizer schedule and early stopping. Warmup (10% of
        # epochs) and the cosine floor (lrf=0.1) are derived smart defaults.
        cos_lr_checkbox = QCheckBox("Warmup → cosine LR schedule")
        cos_lr_checkbox.setChecked(True)
        cos_lr_checkbox.setToolTip(
            "Warmup then cosine decay to a 10% floor (Ultralytics cos_lr, lrf=0.1). "
            "Uncheck to hold the peak learning rate constant."
        )
        layout.addWidget(cos_lr_checkbox)

        lr0_label = QLabel("Peak learning rate (lr0):")
        lr0_input = QDoubleSpinBox()
        lr0_input.setDecimals(5)
        lr0_input.setRange(1e-5, 1.0)
        lr0_input.setSingleStep(1e-3)
        lr0_input.setValue(0.01)
        layout.addWidget(lr0_label)
        layout.addWidget(lr0_input)

        patience_label = QLabel("Early-stop patience (epochs, 0 = off):")
        patience_input = QSpinBox()
        patience_input.setRange(0, 1000)
        patience_input.setValue(20)
        patience_input.setToolTip(
            "Ultralytics patience: stop when val hasn't improved for this many "
            "epochs; best.pt is still the best epoch. 0 disables early stopping."
        )
        layout.addWidget(patience_label)
        layout.addWidget(patience_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                epochs = int(epochs_input.text())
                imgsz = int(imgsz_input.text())
            except ValueError:
                QMessageBox.warning(
                    self.mw, "Invalid Input", "Epochs and image size must be integers."
                )
                return
            train_opts = build_yolo_train_opts(
                epochs,
                cos_lr=cos_lr_checkbox.isChecked(),
                lr0=lr0_input.value(),
                patience=patience_input.value(),
            )
            self.start_training(epochs, imgsz, train_opts)

    def _configure_mlflow(self):
        """Arm Ultralytics' built-in MLflow callback for the next run.

        Tracking is always on, so this unconditionally sets the env vars the
        callback reads and enables the ``mlflow`` Ultralytics setting. Only the
        destination (URI/experiment) is configurable, via Settings → Experiment
        Tracking.
        """
        # Whole body is wrapped: configuring tracking must never abort a run
        # (mirrors MLflowTracker.start()'s blanket crash-safety on the SAM path).
        # A pathological override URI or an Ultralytics import hiccup degrades
        # this run to untracked rather than killing it before it starts.
        try:
            from ..app_settings import load_mlflow_prefs
            from ..training.mlflow_tracker import (
                resolve_tracking_uri,
                to_mlflow_uri,
            )

            _, experiment = load_mlflow_prefs()
            store = resolve_tracking_uri(self.mw)
            # Ultralytics' callback feeds this straight to mlflow.set_tracking_uri,
            # which rejects a bare Windows path — hand it a proper file:// URI.
            os.environ["MLFLOW_TRACKING_URI"] = to_mlflow_uri(store)
            # mlflow 3.x raises on the local file store unless this is set.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
            # NOTE: this persists to the *global* Ultralytics settings file on
            # disk (~/.config/Ultralytics/settings.json), not just process
            # state. Harmless here since we always want it True, but a manual
            # `yolo` run elsewhere will also have mlflow enabled.
            from ultralytics import settings as ultra_settings

            ultra_settings.update({"mlflow": True})
            self.mw.training_dialog.update_info(
                f"MLflow tracking → {store} (experiment '{experiment}')."
            )
        except Exception as exc:
            print(f"Could not configure MLflow tracking: {exc}")
            self.mw.training_dialog.update_info(
                f"MLflow tracking could not be configured ({exc}); "
                "training continues untracked."
            )

    def start_training(self, epochs, imgsz, train_opts=None):
        if not hasattr(self.mw, "training_dialog"):
            self.mw.training_dialog = TrainingInfoDialog(self.mw)
        # Clear last run's log so consecutive runs don't stack (mirrors the SAM
        # fine-tune dialog; otherwise a new run's output appends under the old,
        # making a fresh run look like it resumed mid-way). Issue #35 PR-3.
        self.mw.training_dialog.info_text.clear()
        self.mw.training_dialog.show()
        # Re-show/enable Stop for this run (training_finished hides it).
        self.mw.training_dialog.stop_button.setEnabled(True)
        self.mw.training_dialog.stop_button.setText("Stop Training")
        self.mw.training_dialog.stop_button.show()

        self._configure_mlflow()

        self.mw.yolo_trainer.progress_signal.connect(
            self.mw.training_dialog.update_info
        )
        self.mw.yolo_trainer.mlflow_run_url.connect(self._on_mlflow_run_url)
        self.mw.yolo_trainer.set_progress_callback(self.mw.training_dialog.update_info)
        self.mw.training_dialog.stop_signal.connect(
            self.mw.yolo_trainer.stop_training_signal
        )

        self.mw.training_thread = TrainingThread(
            self.mw.yolo_trainer, epochs, imgsz, train_opts
        )
        self.mw.training_thread.finished.connect(self.training_finished)
        self.mw.training_thread.start()

    def _on_mlflow_run_url(self, url):
        """The YOLO run has opened in MLflow (signalled from the worker thread;
        this runs on the GUI thread). Show a clickable link in the progress
        dialog, start the MLflow UI server once, and open the run in the
        browser. Mirrors SAMTrainController._on_mlflow_run_url; tracking display
        must never disturb the run, so it is best-effort and self-contained."""
        import webbrowser

        from PyQt6.QtCore import QTimer

        from ..training.mlflow_tracker import (
            resolve_tracking_uri,
            start_mlflow_ui_server,
        )

        dlg = getattr(self.mw, "training_dialog", None)
        if dlg is not None:
            dlg.update_info_link("🔗 Open this run in MLflow", url)
        try:
            if not self._mlflow_ui_started:
                ok, _ = start_mlflow_ui_server(
                    resolve_tracking_uri(self.mw),
                    log=dlg.update_info if dlg is not None else None,
                )
                # Latch only on success so a failed launch retries next run.
                self._mlflow_ui_started = ok
                # Give a cold-started server a moment before opening the tab so
                # the browser doesn't land on a connection error. (Non-blocking.)
                QTimer.singleShot(2500 if ok else 0, lambda: webbrowser.open(url))
            else:
                webbrowser.open(url)
        except Exception as exc:
            print(f"Could not open MLflow UI for the run: {exc}")

    def training_finished(self, results):
        # Training is over — hide Stop entirely (only Close remains); the next
        # run re-shows it in start_training.
        self.mw.training_dialog.stop_button.hide()
        self.mw.training_dialog.stop_button.setText("Stop Training")
        # One guard around all three so a partial teardown (double-fire / error
        # path) can't leak a still-connected signal into the next run — mirrors
        # SAMTrainController.training_finished.
        try:
            self.mw.yolo_trainer.progress_signal.disconnect(
                self.mw.training_dialog.update_info
            )
            self.mw.yolo_trainer.mlflow_run_url.disconnect(self._on_mlflow_run_url)
            self.mw.training_dialog.stop_signal.disconnect(
                self.mw.yolo_trainer.stop_training_signal
            )
        except TypeError:
            pass  # already disconnected

        if isinstance(results, str):
            QMessageBox.critical(
                self.mw,
                "Training Error",
                f"An error occurred during training: {results}",
            )
        else:
            saved = getattr(self.mw.yolo_trainer, "last_saved_model_path", None)
            where = (
                f"\n\nSaved to:\n{saved}\n\nIt's now selectable under "
                "Prediction Settings → Load Model."
                if saved
                else ""
            )
            QMessageBox.information(
                self.mw,
                "Training Complete",
                f"YOLO model training completed successfully.{where}",
            )

    def set_confidence_threshold(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        current_threshold = self.mw.yolo_trainer.conf_threshold
        new_threshold, ok = QInputDialog.getDouble(
            self.mw,
            "Set Confidence Threshold",
            "Enter confidence threshold (0-1):",
            current_threshold,
            0,
            1,
            2,
        )
        if ok:
            self.mw.yolo_trainer.set_conf_threshold(new_threshold)
            QMessageBox.information(
                self.mw,
                "Threshold Updated",
                f"Confidence threshold set to {new_threshold}",
            )

    def show_predict_dialog(self):
        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(self.mw, "No Model", "Please load a YOLO model first.")
            return

        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Predict with YOLO Model")
        layout = QVBoxLayout()

        image_list = QListWidget()
        for image_name in self.mw.image_paths.keys():
            image_list.addItem(image_name)
        layout.addWidget(QLabel("Select images for prediction:"))
        layout.addWidget(image_list)

        conf_label = QLabel("Confidence Threshold:")
        conf_input = QDoubleSpinBox()
        conf_input.setRange(0, 1)
        conf_input.setSingleStep(0.01)
        conf_input.setValue(self.mw.yolo_trainer.conf_threshold)
        layout.addWidget(conf_label)
        layout.addWidget(conf_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        predict_button = QPushButton("Predict")
        button_box.addButton(predict_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_images = [item.text() for item in image_list.selectedItems()]
            conf = conf_input.value()
            self.mw.yolo_trainer.set_conf_threshold(conf)
            self.run_predictions(selected_images)

    def run_predictions(self, selected_images):
        for image_name in selected_images:
            image_path = self.mw.image_paths[image_name]
            results = self.mw.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, image_name)

    def predict_single_image(self, file_name):
        if self.mw.is_multi_dimensional(file_name):
            return

        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw,
                "No Model",
                "Please load a YOLO model first from the YOLO > Prediction Settings > Load Model menu.",
            )
            return

        self.mw.deactivate_sam_tools()

        image_path = self.mw.image_paths[file_name]
        try:
            results = self.mw.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, file_name)
        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )

    def process_yolo_results(self, results, image_name):
        image_path = self.mw.image_paths[image_name]
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self.mw, "Error", f"Failed to load image: {image_name}")
            return
        original_height, original_width = image.shape[:2]

        temp_annotations = {}

        try:
            results, input_size, original_size = results
            input_height, input_width = input_size
            orig_height, orig_width = original_size

            scale_x = original_width / orig_width
            scale_y = original_height / orig_height

            # A YOLO checkpoint is exclusively one task — pose models never
            # also emit masks — so this is decided once for all results.
            is_pose = getattr(self.mw.yolo_trainer.model, "task", None) == "pose"

            for result in results:
                boxes = result.boxes

                if is_pose:
                    keypoints = result.keypoints
                    if keypoints is None or len(result.boxes) == 0:
                        continue
                    for kpts, box in zip(keypoints, result.boxes):
                        try:
                            class_id = int(box.cls)
                            class_name = self.mw.yolo_trainer.class_names[class_id]
                            score = float(box.conf)

                            xy = kpts.xy.cpu().numpy()[0]  # (K, 2) pixel coords, Ultralytics orig_img space
                            flat = []
                            for x, y in xy:
                                # Ultralytics gives a per-point presence confidence, not a true
                                # 3-state COCO occlusion signal, and the instance already passed
                                # the box-level conf_threshold gate -- thresholding per point again
                                # would just be noise dressed up as meaningful occlusion data.
                                # Always mark visible (v=2); the user can hand-correct via the
                                # existing right-click-to-toggle-visibility edit gesture on review.
                                flat.extend([float(x) * scale_x, float(y) * scale_y, 2])

                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            bbox = [x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y]

                            temp_class_name = f"Temp-{class_name}"
                            schema = self.mw.yolo_trainer.prediction_keypoint_schema
                            if schema is not None and temp_class_name not in self.mw.keypoint_schemas:
                                self.mw.keypoint_schemas[temp_class_name] = copy.deepcopy(schema)

                            if temp_class_name not in temp_annotations:
                                temp_annotations[temp_class_name] = []
                            temp_annotations[temp_class_name].append({
                                "keypoints": flat,
                                # num_keypoints == K (all points), correct only
                                # because every point is force-stamped v=2 above.
                                # If the "always v=2" simplification is ever
                                # relaxed to emit v=0, this must become the COCO
                                # count of labelled (v>0) points instead.
                                "num_keypoints": len(xy),
                                "bbox": bbox,
                                "category_name": temp_class_name,
                                "score": score,
                                "temp": True,
                            })
                        except IndexError:
                            QMessageBox.warning(
                                self.mw,
                                "Class Mismatch",
                                "There is a mismatch between the model and the YAML file classes. "
                                "Please check that the YAML file corresponds to the loaded model.",
                            )
                            return
                    continue  # a pose result has no masks; nothing else to do for it

                masks = result.masks

                if masks is None:
                    print(f"No masks found for {image_name}")
                    continue

                for mask, box in zip(masks, boxes):
                    try:
                        class_id = int(box.cls)
                        class_name = self.mw.yolo_trainer.class_names[class_id]
                        score = float(box.conf)

                        mask_array = mask.data.cpu().numpy()[0]
                        mask_array = cv2.resize(mask_array, (orig_width, orig_height))
                        contours, _ = cv2.findContours(
                            (mask_array > 0.5).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        if contours:
                            epsilon = 0.005 * cv2.arcLength(contours[0], True)
                            approx = cv2.approxPolyDP(contours[0], epsilon, True)
                            polygon = approx.flatten().tolist()

                            scaled_polygon = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] * scale_x
                                y = polygon[i + 1] * scale_y
                                scaled_polygon.extend([x, y])

                            temp_class_name = f"Temp-{class_name}"
                            if temp_class_name not in temp_annotations:
                                temp_annotations[temp_class_name] = []

                            temp_annotation = {
                                "segmentation": scaled_polygon,
                                "category_name": temp_class_name,
                                "score": score,
                                "temp": True,
                            }
                            temp_annotations[temp_class_name].append(temp_annotation)
                    except IndexError:
                        QMessageBox.warning(
                            self.mw,
                            "Class Mismatch",
                            "There is a mismatch between the model and the YAML file classes. "
                            "Please check that the YAML file corresponds to the loaded model.",
                        )
                        return

        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )
            return

        self.mw.add_temp_classes(temp_annotations)
        self.mw.update_class_list()
        self.mw.image_label.update()

        if temp_annotations:
            total_predictions = sum(len(anns) for anns in temp_annotations.values())
            QMessageBox.information(
                self.mw,
                "Review Predictions",
                f"Found {total_predictions} predictions for {len(temp_annotations)} classes.\n"
                "Use class visibility checkboxes to review.\n"
                "Press Enter to accept or Esc to reject visible predictions.",
            )
        else:
            QMessageBox.information(
                self.mw,
                "No Predictions",
                "No predictions were found for this image.",
            )

        self.mw.deactivate_sam_tools()
