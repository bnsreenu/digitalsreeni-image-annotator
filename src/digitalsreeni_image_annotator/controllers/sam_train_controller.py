"""SAM 2 fine-tuning coordination controller.

Mirrors ``YOLOController``'s shape (menu → validate → config dialog →
``QThread`` worker → finished handler) but drives the Ultralytics-native
:class:`~..training.sam_trainer.SAMFineTuner` instead of ``model.train`` —
Ultralytics has no SAM trainer (see the SAM fine-tuning ADR).

Key differences from YOLO training:
- **GPU gate**: decoder fine-tuning is realistically GPU-only, so a CPU-only
  box is hard-warned before a run starts.
- **Re-entrancy**: the trainer loads its own SAM instance on a worker thread
  (separate from the resident inference model), so this is not a one-model
  race. But two SAM models on one CUDA context is, so the SAM inference UI
  (tools + model selector + this menu) is locked for the duration and the
  trainer never goes through ``sam_utils._run_sync``.
- On success the fine-tuned checkpoint is registered into the SAM model
  selector so it's immediately usable for annotation.
"""

import os

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..dialogs.sam_trainer_dialog import SAMTrainConfigDialog
from ..dialogs.yolo_trainer import TrainingInfoDialog
from ..training.sam_trainer import (
    SAMFineTuner,
    list_custom_models,
    make_custom_filename,
)


class SAMTrainingThread(QThread):
    """Runs a fine-tuning job off the GUI thread. Emits the result dict on
    success or the exception's string on failure (same contract as YOLO's
    ``TrainingThread``)."""

    finished = pyqtSignal(object)

    def __init__(self, trainer: SAMFineTuner, base_model, groups, config):
        super().__init__()
        self.trainer = trainer
        self.base_model = base_model
        self.groups = groups
        self.config = config

    def run(self):
        try:
            result = self.trainer.train(self.base_model, self.groups, **self.config)
            self.finished.emit(result)
        except Exception as e:  # surfaced to the GUI thread by training_finished
            import traceback
            traceback.print_exc()
            self.finished.emit(str(e))


class SAMTrainController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self._mlflow_ui_started = False  # launch the UI server at most once

    # -- menu ----------------------------------------------------------------

    def setup_sam_train_menu(self):
        menu = self.mw.menuBar().addMenu("SAM &Fine-Tune (beta)")
        self._menu = menu

        train_project = QAction("Train on Current Project…", self.mw)
        train_project.triggered.connect(self.train_on_project)
        menu.addAction(train_project)

        prepare = QAction("Prepare SAM Dataset…", self.mw)
        prepare.triggered.connect(self.prepare_dataset)
        menu.addAction(prepare)

        train_folder = QAction("Train from Dataset Folder…", self.mw)
        train_folder.triggered.connect(self.train_from_folder)
        menu.addAction(train_folder)

        menu.addSeparator()
        refresh = QAction("Refresh Fine-Tuned Model List", self.mw)
        refresh.triggered.connect(self.refresh_model_selector)
        menu.addAction(refresh)

    # -- entry points --------------------------------------------------------

    def train_on_project(self):
        from ..training.sam_dataset import build_groups_from_project

        groups = build_groups_from_project(
            self.mw.all_annotations,
            self.mw.image_paths,
            self.mw.slices,
            self.mw.image_slices,
        )
        if not groups:
            QMessageBox.warning(
                self.mw, "No Training Data",
                "No usable annotations found. Annotate some objects (polygons "
                "or boxes) first.",
            )
            return
        self._launch(groups)

    def prepare_dataset(self):
        from ..io.export_formats import export_sam_dataset

        out_dir = QFileDialog.getExistingDirectory(self.mw, "Choose dataset output folder")
        if not out_dir:
            return
        try:
            _, manifest = export_sam_dataset(
                self.mw.all_annotations,
                self.mw.class_mapping,
                self.mw.image_paths,
                self.mw.slices,
                self.mw.image_slices,
                out_dir,
            )
        except Exception as e:
            QMessageBox.critical(self.mw, "Export Failed", str(e))
            return
        QMessageBox.information(
            self.mw, "Dataset Prepared",
            f"SAM dataset written to:\n{manifest}\n\nUse 'Train from Dataset "
            f"Folder…' to fine-tune on it.",
        )

    def train_from_folder(self):
        from ..training.sam_dataset import build_groups_from_folder

        folder = QFileDialog.getExistingDirectory(self.mw, "Choose prepared SAM dataset folder")
        if not folder:
            return
        try:
            groups = build_groups_from_folder(folder)
        except Exception as e:
            QMessageBox.critical(self.mw, "Load Failed", str(e))
            return
        if not groups:
            QMessageBox.warning(self.mw, "Empty Dataset", "No usable entries in that folder.")
            return
        self._launch(groups)

    # -- run -----------------------------------------------------------------

    def _launch(self, groups):
        if hasattr(self.mw, "sam_training_thread") and self.mw.sam_training_thread is not None \
                and self.mw.sam_training_thread.isRunning():
            QMessageBox.information(self.mw, "Training Busy", "A fine-tuning run is already in progress.")
            return
        if not self._gpu_gate():
            return

        dialog = SAMTrainConfigDialog(self.mw)
        if dialog.exec() != SAMTrainConfigDialog.DialogCode.Accepted:
            return
        cfg = dialog.get_config()
        base_model = cfg.pop("base_model")
        out_name = cfg.pop("out_name")
        cfg["out_path"] = make_custom_filename(base_model, out_name)

        # MLflow tracking (issue #74) — always on. Build a tracker now but
        # start the run inside train() on the worker thread (MLflow runs are
        # thread-bound). Only the destination (URI/experiment) is configurable,
        # via Settings → Experiment Tracking.
        from ..app_settings import load_mlflow_prefs
        from ..training.mlflow_tracker import MLflowTracker, resolve_tracking_uri
        _, experiment = load_mlflow_prefs()
        # No log callback here: the trainer wires the tracker's log to its own
        # thread-safe ``progress_signal`` so tracker messages reach the GUI
        # via a queued connection (never a direct cross-thread QTextEdit write).
        cfg["tracker"] = MLflowTracker(
            tracking_uri=resolve_tracking_uri(self.mw),
            experiment_name=experiment,
            run_name=out_name,
        )

        # The trainer loads its OWN SAM instance on a worker thread. The
        # resident inference model is separate, but it stays reachable from the
        # GUI — running inference (its own CUDA work) alongside training on the
        # same device/context invites OOM and contention. Deactivate the tools
        # and lock the SAM inference UI for the duration so no concurrent
        # inference or model-swap can be triggered.
        self.mw.sam_controller.deactivate_sam_tools()
        self._set_sam_ui_locked(True)

        # Everything from here to start() must restore the UI if it raises —
        # otherwise training_finished (the only other unlock site) never fires
        # and the SAM tools stay disabled until app restart.
        try:
            self.mw.sam_finetuner = SAMFineTuner()
            if not hasattr(self.mw, "sam_training_dialog"):
                self.mw.sam_training_dialog = TrainingInfoDialog(self.mw)
            self.mw.sam_training_dialog.setWindowTitle("SAM Fine-Tuning Progress")
            # The dialog is reused across runs (same TrainingInfoDialog class as
            # YOLO, but a separate instance) — clear the previous run's log so a
            # new run doesn't append under stale output.
            self.mw.sam_training_dialog.info_text.clear()
            self.mw.sam_training_dialog.stop_button.setEnabled(True)
            self.mw.sam_training_dialog.stop_button.setText("Stop Training")
            self.mw.sam_training_dialog.stop_button.show()
            self.mw.sam_training_dialog.show()

            self.mw.sam_finetuner.progress_signal.connect(self.mw.sam_training_dialog.update_info)
            self.mw.sam_finetuner.mlflow_run_url.connect(self._on_mlflow_run_url)
            self.mw.sam_training_dialog.stop_signal.connect(self.mw.sam_finetuner.stop_training_signal)

            self.mw.sam_training_thread = SAMTrainingThread(
                self.mw.sam_finetuner, base_model, groups, cfg
            )
            self.mw.sam_training_thread.finished.connect(self.training_finished)
            self.mw.sam_training_thread.start()
        except Exception as e:
            self._set_sam_ui_locked(False)
            QMessageBox.critical(self.mw, "Could Not Start Training", str(e))

    def _on_mlflow_run_url(self, url):
        """The fine-tuning run has opened in MLflow (signalled from the worker
        thread; this runs on the GUI thread). Show a clickable link in the
        progress dialog, start the MLflow UI server once, and open the run in
        the browser. Tracking display must never disturb the run, so this is
        best-effort and swallows its own errors."""
        import webbrowser

        from PyQt6.QtCore import QTimer

        from ..training.mlflow_tracker import (
            resolve_tracking_uri,
            start_mlflow_ui_server,
        )

        dlg = getattr(self.mw, "sam_training_dialog", None)
        if dlg is not None:
            dlg.update_info_link("🔗 Open this run in MLflow", url)
        try:
            if not self._mlflow_ui_started:
                ok, _ = start_mlflow_ui_server(
                    resolve_tracking_uri(self.mw),
                    log=dlg.update_info if dlg is not None else None,
                )
                # Only latch the guard on success, so a failed launch retries
                # on the next run rather than disabling the UI for the session.
                self._mlflow_ui_started = ok
                # Give a cold-started server a moment before opening the tab so
                # the browser doesn't land on a connection error. (Non-blocking.)
                QTimer.singleShot(2500 if ok else 0, lambda: webbrowser.open(url))
            else:
                webbrowser.open(url)
        except Exception as exc:
            print(f"Could not open MLflow UI for the run: {exc}")

    def _gpu_gate(self) -> bool:
        """Warn (and let the user back out) when no usable GPU is present."""
        from ..core.torch_utils import resolve_torch_device

        device, _ = resolve_torch_device()
        if device == "cuda":
            return True
        choice = QMessageBox.warning(
            self.mw, "No GPU — training will be very slow",
            "No usable CUDA GPU was detected. Fine-tuning SAM on CPU is "
            "impractically slow (minutes per image). Continue anyway with a "
            "small run?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return choice == QMessageBox.StandardButton.Yes

    def _set_sam_ui_locked(self, locked: bool):
        """Disable/enable SAM inference controls + the fine-tune menu so no
        concurrent inference, model swap, or second training run can start
        while a run is in flight."""
        for attr in ("sam_box_button", "sam_points_button", "sam_model_selector"):
            widget = getattr(self.mw, attr, None)
            if widget is not None:
                widget.setEnabled(not locked)
        if getattr(self, "_menu", None) is not None:
            self._menu.setEnabled(not locked)

    def training_finished(self, result):
        self._set_sam_ui_locked(False)
        dlg = self.mw.sam_training_dialog
        # Training is over — HIDE Stop entirely (only Close remains). _launch
        # re-shows and re-enables it for the next run.
        dlg.stop_button.hide()
        dlg.stop_button.setText("Stop Training")
        try:
            self.mw.sam_finetuner.progress_signal.disconnect(dlg.update_info)
            self.mw.sam_finetuner.mlflow_run_url.disconnect(self._on_mlflow_run_url)
            dlg.stop_signal.disconnect(self.mw.sam_finetuner.stop_training_signal)
        except TypeError:
            pass  # already disconnected

        if isinstance(result, str):
            QMessageBox.critical(self.mw, "Fine-Tuning Error", f"An error occurred:\n{result}")
            return

        self.refresh_model_selector()
        out_path = result.get("out_path")
        display = f"★ {os.path.splitext(os.path.basename(out_path))[0]}" if out_path else None
        if display and hasattr(self.mw, "sam_model_selector"):
            idx = self.mw.sam_model_selector.findText(display)
            if idx >= 0:
                self.mw.sam_model_selector.setCurrentIndex(idx)
        QMessageBox.information(
            self.mw, "Fine-Tuning Complete",
            f"Saved and verified:\n{out_path}\n\nSelected it in the SAM model "
            f"dropdown — use SAM-box / SAM-points to try it.",
        )

    # -- selector ------------------------------------------------------------

    def refresh_model_selector(self):
        """Register fine-tuned checkpoints and (re)add them to the SAM dropdown."""
        customs = list_custom_models()
        self.mw.sam_utils.register_custom_models(customs)
        selector = getattr(self.mw, "sam_model_selector", None)
        if selector is None:
            return
        existing = {selector.itemText(i) for i in range(selector.count())}
        for display in customs:
            if display not in existing:
                selector.addItem(display)
