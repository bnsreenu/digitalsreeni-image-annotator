import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QLabel, QDialogButtonBox, QComboBox)
import yaml
import numpy as np
from pathlib import Path
from ..io.export_formats import export_yolo_v5plus
from ..utils import models_base_dir
from ..core.keypoint_schema import sanitize_schema


from collections import deque


from PyQt6.QtWidgets import QTextBrowser
from PyQt6.QtGui import QPalette, QTextCharFormat, QTextCursor
from PyQt6.QtCore import pyqtSignal, QObject

from ..core.logging_config import get_logger

logger = get_logger(__name__)


# Trained YOLO checkpoints land here, namespaced per run, mirroring SAM's
# ``models/sam/custom`` layout (see training.sam_trainer.SAM_CUSTOM_DIR). Each
# run is its own ``<name>/`` sub-dir holding Ultralytics' ``weights/best.pt``
# plus a ``data.yaml`` (class names) so the prediction loader has both halves
# it needs without the user hunting through ``runs/``.
YOLO_MODELS_DIR = os.path.join(models_base_dir(), "yolo")
YOLO_CUSTOM_DIR = os.path.join(YOLO_MODELS_DIR, "custom")


def _sanitize_run_name(name):
    """Filesystem-safe run name; falls back to ``model`` when empty."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")
    return safe or "model"


def _reclaim_gpu():
    """Best-effort GPU/host reclaim between training runs.

    Ultralytics keeps a model<->trainer reference cycle that pins CUDA
    tensors, so dropping the Python reference alone leaves a stale copy on
    the device until cyclic GC happens to run (see CLAUDE.md "Releasing
    Model GPU Memory"). Force a collect + empty_cache so a reload doesn't
    stack a second copy on the GPU. Never raises — reclaim is best-effort.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def list_custom_yolo_models() -> dict:
    """``{display_name: (model_path, yaml_path)}`` for trained YOLO models.

    Scans ``YOLO_CUSTOM_DIR`` for run folders that produced a ``best.pt``; the
    sibling ``data.yaml`` (written at train time) supplies the class names the
    prediction loader validates against. Used to populate the Load-Model
    dialog's dropdown so a freshly trained model is selectable without browsing.
    """
    out = {}
    if os.path.isdir(YOLO_CUSTOM_DIR):
        for entry in sorted(os.listdir(YOLO_CUSTOM_DIR)):
            run_dir = os.path.join(YOLO_CUSTOM_DIR, entry)
            best = os.path.join(run_dir, "weights", "best.pt")
            if os.path.isfile(best):
                yml = os.path.join(run_dir, "data.yaml")
                out[f"★ {entry}"] = (best, yml if os.path.isfile(yml) else "")
    return out


def _resolve_training_yaml(yaml_dir, yaml_content):
    """Resolve a dataset yaml's train/val pointers to absolute paths for training.

    Honors the yaml's own train/val pointers — ``export_yolo_v5plus`` points
    ``val`` at ``images/val`` when a held-out set was routed there, and falls
    back to ``images/train`` when it wasn't (``val_split=0`` or a single-image
    project) — so the user's split is kept while YOLO is never fed an empty val
    dir. Returns a new dict; the input is not mutated.

    Invariant: the dataset root is the yaml's own directory (``yaml_dir``).
    Every yaml this trainer consumes satisfies it — the ones ``prepare_dataset``
    generates have ``path == output_dir == yaml_dir``, and ``load_yaml``
    relativizes train/val to the yaml's own directory. The standalone ``path``
    key is therefore redundant once train/val are absolute, and is dropped.
    """
    resolved = dict(yaml_content)
    train_rel = resolved.get("train", os.path.join("images", "train"))
    val_rel = resolved.get("val", train_rel)  # export fallback ⇒ never empty
    resolved["train"] = str((Path(yaml_dir) / train_rel).resolve())
    resolved["val"] = str((Path(yaml_dir) / val_rel).resolve())
    resolved.pop("path", None)
    return resolved


class TrainingInfoDialog(QDialog):
    stop_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Progress")
        self.setModal(False)
        self.layout = QVBoxLayout(self)

        # QTextBrowser (not QTextEdit) so the MLflow run link rendered via
        # update_info_link() is clickable; setOpenExternalLinks opens it in the
        # system browser instead of trying to navigate inside the widget.
        self.info_text = QTextBrowser(self)
        self.info_text.setReadOnly(True)
        self.info_text.setOpenExternalLinks(True)
        self.layout.addWidget(self.info_text)

        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.clicked.connect(self.stop_training)
        self.layout.addWidget(self.stop_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.hide)
        self.layout.addWidget(self.close_button)

        self.setMinimumSize(400, 300)

    def _append_block(self, text, char_format):
        """Append one line with an explicit char format.

        Each line carries its own format, so the anchor format of a link line
        can't bleed into the plain lines appended after it (which would render
        later progress rows as clickable links — they aren't). Using a cursor
        with an explicit format is what makes that guarantee; ``append()`` reuses
        the document's trailing format and leaks the anchor.
        """
        doc = self.info_text.document()
        cursor = QTextCursor(doc)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not doc.isEmpty():
            cursor.insertBlock()
        cursor.insertText(text, char_format)
        bar = self.info_text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def update_info(self, text):
        self._append_block(text, QTextCharFormat())  # plain — never a link

    def update_info_link(self, label, url):
        """Append a clickable link (opens in the system browser)."""
        fmt = QTextCharFormat()
        fmt.setAnchor(True)
        fmt.setAnchorHref(url)
        fmt.setFontUnderline(True)
        # Theme-correct link colour from the palette (No Hardcoded Colors Rule).
        fmt.setForeground(self.info_text.palette().color(QPalette.ColorRole.Link))
        self._append_block(label, fmt)

    def stop_training(self):
        self.stop_signal.emit()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping...")

    def closeEvent(self, event):
        event.ignore()
        self.hide()

class LoadPredictionModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Prediction Model and YAML")
        self.model_path = ""
        self.yaml_path = ""

        layout = QVBoxLayout(self)

        # Trained-models dropdown — auto-discovered from models/yolo/custom so a
        # freshly trained run is one click away (browsing below stays available
        # for external models). Maps each entry to its (model, yaml) pair.
        self._trained = list_custom_yolo_models()
        if self._trained:
            trained_layout = QHBoxLayout()
            self.trained_combo = QComboBox()
            self.trained_combo.addItem("— select a trained model —", None)
            for display, paths in self._trained.items():
                self.trained_combo.addItem(display, paths)
            self.trained_combo.currentIndexChanged.connect(self._on_trained_selected)
            trained_layout.addWidget(QLabel("Trained Model:"))
            trained_layout.addWidget(self.trained_combo)
            layout.addLayout(trained_layout)

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        model_button = QPushButton("Browse")
        model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(QLabel("Model File:"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_button)
        layout.addLayout(model_layout)

        # YAML file selection
        yaml_layout = QHBoxLayout()
        self.yaml_edit = QLineEdit()
        yaml_button = QPushButton("Browse")
        yaml_button.clicked.connect(self.browse_yaml)
        yaml_layout.addWidget(QLabel("YAML File:"))
        yaml_layout.addWidget(self.yaml_edit)
        yaml_layout.addWidget(yaml_button)
        layout.addLayout(yaml_layout)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if file_name:
            self.model_path = file_name
            self.model_edit.setText(file_name)

    def browse_yaml(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml *.yml)")
        if file_name:
            self.yaml_path = file_name
            self.yaml_edit.setText(file_name)

    def _on_trained_selected(self, _index):
        """Fill model + yaml from the chosen trained run (placeholder clears)."""
        paths = self.trained_combo.currentData()
        if not paths:
            return
        model_path, yaml_path = paths
        self.model_path = model_path
        self.model_edit.setText(model_path)
        self.yaml_path = yaml_path
        self.yaml_edit.setText(yaml_path)

class YOLOTrainer(QObject):
    progress_signal = pyqtSignal(str)
    # Emitted once per run with the MLflow UI deep link, mirroring the SAM
    # fine-tuner's signal of the same name so the controller can show a
    # clickable link / open the run (see YOLOController._on_mlflow_run_url).
    mlflow_run_url = pyqtSignal(str)

    def __init__(self, project_dir, main_window):
        super().__init__()
        self.project_dir = project_dir
        self.main_window = main_window
        self.model = None
        self.dataset_path = os.path.join(project_dir, "yolo_dataset")
        self.model_path = os.path.join(project_dir, "yolo_model")
        self.yaml_path = None
        self.yaml_data = None
        self.epoch_info = deque(maxlen=10)
        self.progress_callback = None
        self.total_epochs = None
        self.conf_threshold = 0.25
        self.stop_training = False
        self.class_names = None
        # Reconstructed keypoint schema for the loaded prediction model (#35
        # PR-3), set in load_prediction_model(); always present (None until a
        # prediction model is loaded) so callers never need a hasattr guard.
        self.prediction_keypoint_schema = None
        # Set by train_model() once a run finishes, so the controller can tell
        # the user where the checkpoint landed and offer it for prediction.
        self.last_saved_model_path = None
        self.last_saved_yaml_path = None
        # Path the training model was loaded from, so train_model() can reload a
        # pristine object per run (Ultralytics mutates a YOLO object's overrides
        # during train(), breaking a second train() on the same instance).
        self.loaded_model_path = None
        # Latched so the per-epoch callback emits the MLflow link only once.
        self._mlflow_url_emitted = False

    def load_model(self, model_path=None):
        from ultralytics import YOLO

        if model_path is None:
            model_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Model", "", "YOLO Model (*.pt)")
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.loaded_model_path = model_path
                return True
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error Loading Model", f"Could not load the model. Error: {str(e)}")
        return False

    def prepare_dataset(self, val_split=20):
        output_dir, yaml_path = export_yolo_v5plus(
            self.main_window.all_annotations,
            self.main_window.class_mapping,
            self.main_window.image_paths,
            self.main_window.slices,
            self.main_window.image_slices,
            self.dataset_path,
            val_split,
            keypoint_schemas=self.main_window.keypoint_schemas,
        )

        yaml_path = Path(yaml_path)
        with yaml_path.open('r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)

        # export_yolo_v5plus already writes the correct relative train/val
        # pointers (val falls back to train when no val images were routed),
        # so don't clobber them here — just add the test placeholder.
        yaml_content.setdefault('test', '../test/images')
        
        with yaml_path.open('w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.yaml_path = str(yaml_path)
        return self.yaml_path

    def load_yaml(self, yaml_path=None):
        if yaml_path is None:
            yaml_path, _ = QFileDialog.getOpenFileName(self.main_window, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)")
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                try:
                    yaml_data = yaml.safe_load(f)
                    logger.debug(f"Loaded YAML contents: {yaml_data}")
    
                    # Ensure paths are relative
                    for key in ['train', 'val', 'test']:
                        if key in yaml_data and os.path.isabs(yaml_data[key]):
                            yaml_data[key] = os.path.relpath(yaml_data[key], start=os.path.dirname(yaml_path))
    
                    logger.debug(f"Updated YAML contents: {yaml_data}")
    
                    # Save the updated YAML data
                    self.yaml_data = yaml_data
                    self.yaml_path = yaml_path
    
                    # Write the updated YAML back to the file
                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        yaml.dump(yaml_data, f, default_flow_style=False)
    
                    return True
                except yaml.YAMLError as e:
                    QMessageBox.critical(self.main_window, "Error Loading YAML", f"Invalid YAML file. Error: {str(e)}")
        return False

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1  # Add 1 to start from 1 instead of 0
        total_epochs = trainer.epochs
        loss = trainer.loss.item()
        progress_text = f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"

        # Only emit the signal, don't call the callback directly
        self.progress_signal.emit(progress_text)
        self._emit_mlflow_url()

        if self.stop_training:
            trainer.model.stop = True
            self.stop_training = False
            return False
        return True

    def on_fit_epoch_end(self, trainer):
        """Surface validation loss + mAP + LR once per epoch (after validation).

        ``on_fit_epoch_end`` fires after the epoch's validation pass, so
        ``trainer.metrics`` holds the val losses and mAP. Keys vary by task
        (detect vs segment) and Ultralytics version, so every read is defensive —
        a missing key or format change must never disturb the run. MLflow already
        records these natively via Ultralytics' own callback; this only mirrors
        them into the progress window alongside the train-loss line."""
        try:
            epoch = trainer.epoch + 1
            total = trainer.epochs
            metrics = getattr(trainer, "metrics", None) or {}
            parts = [f"Epoch {epoch}/{total}"]

            # box_loss is always present (detection head); seg_loss only for
            # segmentation runs. Keys here carry no special chars, so match exact.
            for key, label in (("val/box_loss", "val_box_loss"),
                               ("val/seg_loss", "val_seg_loss"),
                               ("val/pose_loss", "val_pose_loss"),
                               ("val/kobj_loss", "val_kobj_loss")):
                val_loss = metrics.get(key)
                if val_loss is not None:
                    parts.append(f"{label}={float(val_loss):.4f}")

            # mAP keys carry a task suffix — "(B)"/"(M)" in Ultralytics' in-memory
            # dict, but some logging layers strip the parens to "B"/"M" (seen in
            # the MLflow store). Match by substring so either form is surfaced
            # (and both detection and segmentation runs work).
            def _first(needle, exclude=None):
                for key, value in metrics.items():
                    if needle in key and (exclude is None or exclude not in key):
                        return value
                return None

            map50 = _first("mAP50", exclude="mAP50-95")
            if map50 is not None:
                parts.append(f"mAP50={float(map50):.4f}")
            map5095 = _first("mAP50-95")
            if map5095 is not None:
                parts.append(f"mAP50-95={float(map5095):.4f}")

            lr = getattr(trainer, "lr", None)
            if isinstance(lr, dict) and lr:
                parts.append(f"lr={float(next(iter(lr.values()))):.2e}")

            if len(parts) > 1:  # something beyond the epoch tag was available
                self.progress_signal.emit("  ".join(parts))
        except Exception:
            logger.exception("Could not surface YOLO val metrics")

    def _emit_mlflow_url(self):
        """Emit the MLflow run's UI deep link, once per run.

        Ultralytics' native MLflow callback starts the run in
        ``on_pretrain_routine_end`` (before any epoch), so by the first
        ``on_train_epoch_end`` ``mlflow.active_run()`` is populated regardless
        of callback ordering. We read its run/experiment ids and hand the
        controller the same deep link the SAM path builds. Best-effort: a
        tracking hiccup must never disturb the run."""
        if self._mlflow_url_emitted:
            return
        try:
            import mlflow

            run = mlflow.active_run()
            if run is None:
                return
            from ..training.mlflow_tracker import run_ui_url

            self._mlflow_url_emitted = True
            self.mlflow_run_url.emit(
                run_ui_url(run.info.experiment_id, run.info.run_id)
            )
        except Exception:
            logger.exception("Could not resolve MLflow run link for YOLO training")
    
    def train_model(self, epochs=100, imgsz=640, *, cos_lr=True, lr0=0.01,
                    lrf=0.1, warmup_epochs=None, patience=20):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        if self.yaml_path is None or not Path(self.yaml_path).exists():
            raise FileNotFoundError("Dataset YAML not found. Please prepare or load a dataset first.")

        # Reload a pristine model from the loaded checkpoint before every run.
        # Ultralytics mutates the YOLO object's `overrides` during train() (it
        # drops the 'model' key), so a SECOND train() on the same instance
        # raises KeyError('model') at engine/model.py. Reloading also clears any
        # accumulated trainer state / stacked callbacks, so each run is
        # independent and reproducible (fine-tunes the loaded pre-trained model,
        # not the previous run's output). See issue #35 PR-3 manual-testing fix.
        if self.loaded_model_path:
            from ultralytics import YOLO

            # Drop the previous object and reclaim before reloading — the old
            # model<->trainer cycle pins CUDA tensors, so clearing the ref alone
            # shows no GPU drop (CLAUDE.md "Releasing Model GPU Memory").
            self.model = None
            _reclaim_gpu()
            self.model = YOLO(self.loaded_model_path)

        # Warmup over the first ~10% of epochs (the issue's recipe) unless the
        # caller overrode it. Floor (lrf) defaults to 10% of the peak LR.
        if warmup_epochs is None:
            warmup_epochs = max(1, round(0.1 * epochs))

        self.stop_training = False
        self.total_epochs = epochs
        self.epoch_info.clear()
        self._mlflow_url_emitted = False
        self.last_saved_model_path = None
        self.last_saved_yaml_path = None

        # Per-epoch progress: on_train_epoch_end has the train loss; on_fit_epoch_end
        # fires AFTER validation so trainer.metrics carries val loss + mAP.
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        self.model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

        try:
            yaml_path = Path(self.yaml_path)
            yaml_dir = yaml_path.parent

            logger.debug(f"Training with YAML: {yaml_path}")
            logger.debug(f"YAML directory: {yaml_dir}")

            with yaml_path.open('r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            logger.debug(f"YAML content: {yaml_content}")

            # Resolve train/val to absolute paths, honoring the split the export
            # wrote (see _resolve_training_yaml for the data-root invariant).
            yaml_content = _resolve_training_yaml(yaml_dir, yaml_content)

            model_task = getattr(self.model, "task", None)
            yaml_is_pose = "kpt_shape" in yaml_content
            if model_task == "pose" and not yaml_is_pose:
                raise ValueError(
                    "The loaded model is a pose model, but the prepared dataset has no "
                    "keypoint annotations (no kpt_shape in data.yaml). Annotate at least "
                    "one pose instance and re-run 'Prepare YOLO Dataset', or load a "
                    "detection/segmentation model instead."
                )
            if model_task != "pose" and yaml_is_pose:
                raise ValueError(
                    "The prepared dataset contains keypoint (pose) annotations, but the "
                    "loaded model is not a pose model. Load a '*-pose.pt' checkpoint (or "
                    "a previously trained pose model), or re-prepare the dataset from a "
                    "project without pose classes."
                )

            # Create the val directory structure if it doesn't exist
            val_img_dir = yaml_dir / 'images' / 'val'
            val_label_dir = yaml_dir / 'labels' / 'val'
            val_img_dir.mkdir(parents=True, exist_ok=True)
            val_label_dir.mkdir(parents=True, exist_ok=True)

            # Write updated YAML with adjusted paths
            temp_yaml_path = yaml_dir / 'temp_train.yaml'
            with temp_yaml_path.open('w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

            logger.debug(f"Training with updated YAML: {temp_yaml_path}")
            logger.debug(f"Updated YAML content: {yaml_content}")

            from ..core.torch_utils import resolve_torch_device
            device, _ = resolve_torch_device()
            # Route the run into models/yolo/custom/<project>/ (Ultralytics
            # auto-increments on collision) instead of the default ./runs, so
            # the checkpoint has a predictable home like SAM's custom dir.
            run_name = _sanitize_run_name(os.path.basename(self.project_dir))
            results = self.model.train(
                data=str(temp_yaml_path), epochs=epochs, imgsz=imgsz, device=device,
                project=YOLO_CUSTOM_DIR, name=run_name,
                cos_lr=cos_lr, lr0=lr0, lrf=lrf, warmup_epochs=warmup_epochs,
                patience=patience,
            )
            self._register_trained_model()
            return results
        finally:
            # Clear our callbacks so they don't stack on a second run (the run's
            # own trainer holds Ultralytics' native callbacks, so this only drops
            # ours).
            self.model.callbacks["on_train_epoch_end"] = []
            self.model.callbacks["on_fit_epoch_end"] = []
            # Remove temporary YAML file
            if 'temp_yaml_path' in locals():
                temp_yaml_path.unlink(missing_ok=True)

    def _register_trained_model(self):
        """Record the run's best.pt + a class-names yaml for later prediction.

        Ultralytics writes ``<save_dir>/weights/best.pt``; we drop a sibling
        ``data.yaml`` carrying the trained class names so the prediction loader
        (which needs both a model and a names yaml) can pick this run up from
        the Load-Model dropdown without the user supplying anything. Best-effort
        — a failure here just leaves the model un-registered, not the run
        broken."""
        try:
            trainer = getattr(self.model, "trainer", None)
            # Guard the empty string: Path("") is Path(".") which .exists() as
            # the cwd, so an absent trainer.best must fall through to save_dir.
            best_attr = getattr(trainer, "best", "") if trainer else ""
            best = Path(best_attr) if best_attr else None
            if not best or not best.exists():
                save_dir = getattr(trainer, "save_dir", None) if trainer else None
                if save_dir:
                    best = Path(save_dir) / "weights" / "best.pt"
            if not best or not best.exists():
                logger.warning("Trained YOLO checkpoint not found; skipping registration.")
                return
            names = self.model.names  # {idx: name} from the trained model
            # Names-only on purpose: load_prediction_model reads `names` only,
            # and this yaml describes a *trained model*, not a dataset — it must
            # NOT carry train/val/path pointers (they'd be stale the moment the
            # dataset moves). Don't "helpfully" add them.
            yaml_out_data = {"names": names, "nc": len(names)}
            # Carry pose metadata (issue #35 PR-3) so a later prediction load can
            # reconstruct a keypoint schema. Prefer the real, hand-authored schema
            # (richer than YOLO-pose's bare K/flip_idx) when every trained class
            # shares one identical schema in this session's main_window.keypoint_schemas;
            # always fall back to bare kpt_shape/flip_idx so an externally-loaded pose
            # yaml still degrades gracefully at prediction-load time.
            try:
                if self.yaml_path and Path(self.yaml_path).exists():
                    with open(self.yaml_path, 'r', encoding='utf-8') as f:
                        train_yaml = yaml.safe_load(f) or {}
                    kpt_shape = train_yaml.get('kpt_shape')
                    if kpt_shape:
                        yaml_out_data['kpt_shape'] = kpt_shape
                        yaml_out_data['flip_idx'] = train_yaml.get('flip_idx')
                        schemas = [self.main_window.keypoint_schemas.get(n) for n in names.values()]
                        if schemas and all(s is not None for s in schemas) and all(s == schemas[0] for s in schemas):
                            yaml_out_data['keypoint_schema'] = schemas[0]
            except Exception:
                logger.exception("Could not carry pose metadata into registered model yaml")
            yaml_out = best.parent.parent / "data.yaml"
            with yaml_out.open("w", encoding="utf-8") as f:
                yaml.dump(yaml_out_data, f, default_flow_style=False)
            self.last_saved_model_path = str(best)
            self.last_saved_yaml_path = str(yaml_out)
            logger.info(f"Trained YOLO model registered: {best}")
            self._prune_run_artifacts(best.parent.parent, best, yaml_out)
        except Exception:
            logger.exception("Could not register trained YOLO model")

    def _prune_run_artifacts(self, run_dir, best, data_yaml):
        """Trim the run dir to the minimum needed to run predictions.

        Ultralytics writes a full run directory (last.pt, args.yaml, results.csv,
        and a dozen plot/mosaic PNG/JPGs). Its native MLflow callback already
        logs *all* of it into the MLflow run (``on_train_end`` →
        ``log_artifact`` over the weights dir + every png/jpg/csv/pt/yaml), so
        once tracking is confirmed those local copies are redundant — keep only
        ``best.pt`` + the names ``data.yaml`` the app needs, and let MLflow be
        the home for the diagnostics.

        Guarded on ``_mlflow_url_emitted``: if the run was *not* MLflow-tracked
        (tracking degraded), the diagnostics live nowhere else, so we keep the
        whole folder rather than destroy them. Best-effort — a cleanup failure
        never breaks the (already finished) run."""
        if not self._mlflow_url_emitted:
            logger.warning("MLflow run not confirmed; keeping full YOLO run dir locally.")
            return
        try:
            keep = {best.resolve(), data_yaml.resolve()}
            for path in run_dir.rglob("*"):
                if path.is_file() and path.resolve() not in keep:
                    path.unlink(missing_ok=True)
            # Drop any now-empty subdirs (e.g. plots/), but weights/ survives
            # because best.pt is still in it.
            for path in sorted(run_dir.rglob("*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            logger.info(f"Pruned YOLO run dir to best.pt + data.yaml (diagnostics in MLflow): {run_dir}")
        except Exception:
            logger.exception("Could not prune YOLO run artifacts")

    def verify_dataset_structure(self):
        yaml_path = Path(self.yaml_path)
        yaml_dir = yaml_path.parent
        
        with yaml_path.open('r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
        
        # Use paths from YAML content
        train_images_dir = yaml_dir / yaml_content.get('train', 'images/train')
        val_images_dir = yaml_dir / yaml_content.get('val', 'images/val')
        train_labels_dir = yaml_dir / 'labels' / 'train'  # Labels directory corresponds to images
        val_labels_dir = yaml_dir / 'labels' / 'val'      # Labels directory corresponds to images
        
        # Check both train and val directories
        missing_dirs = []
        if not train_images_dir.exists():
            missing_dirs.append(f"Training images directory: {train_images_dir}")
        if not train_labels_dir.exists():
            missing_dirs.append(f"Training labels directory: {train_labels_dir}")
        if not val_images_dir.exists():
            missing_dirs.append(f"Validation images directory: {val_images_dir}")
        if not val_labels_dir.exists():
            missing_dirs.append(f"Validation labels directory: {val_labels_dir}")
        
        if missing_dirs:
            raise FileNotFoundError("The following directories were not found:\n" + "\n".join(missing_dirs))
        
        logger.debug("Dataset structure verified:")
        logger.debug(f"Train images: {train_images_dir}")
        logger.debug(f"Train labels: {train_labels_dir}")
        logger.debug(f"Val images: {val_images_dir}")
        logger.debug(f"Val labels: {val_labels_dir}")

    def check_ultralytics_settings(self):
        settings_path = Path.home() / ".config" / "Ultralytics" / "settings.yaml"
        if settings_path.exists():
            with settings_path.open('r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            logger.debug(f"Ultralytics settings: {settings}")
        else:
            logger.warning("Ultralytics settings file not found.")
            
    def stop_training_signal(self):
        self.stop_training = True
        self.progress_signal.emit("Stopping training...")

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    
    def stop_training_callback(self, trainer):
        if getattr(self, 'stop_training', False):
            trainer.model.stop = True
            self.stop_training = False
            

            
    def on_epoch_end(self, trainer):
        # Get current epoch
        epoch = trainer.epoch if hasattr(trainer, 'epoch') else trainer.current_epoch

        # Get total epochs
        total_epochs = self.total_epochs  # Use the value we set in train_model

        # Get loss
        if hasattr(trainer, 'metrics') and 'train/box_loss' in trainer.metrics:
            loss = trainer.metrics['train/box_loss']
        elif hasattr(trainer, 'loss'):
            loss = trainer.loss
        else:
            loss = 0  # Default value if loss can't be found

        # Ensure loss is a number
        loss = float(loss)

        info = f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"
        self.epoch_info.append(info)
        
        display_text = "Current Progress:\n" + "\n".join(self.epoch_info)
        if self.progress_callback:
            self.progress_callback(display_text)


    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        save_path, _ = QFileDialog.getSaveFileName(self.main_window, "Save YOLO Model", "", "YOLO Model (*.pt)")
        if save_path:
            self.model.save(save_path)
            return True
        return False

    def load_prediction_model(self, model_path, yaml_path):
        from ultralytics import YOLO

        try:
            self.model = YOLO(model_path)
            # Keep loaded_model_path in sync with EVERY self.model assignment
            # (load_model sets it too) so train_model's reload trains the model
            # the user actually loaded last — loading a trained best.pt here and
            # then hitting Train Model continues from it, not the stale original
            # pretrained. Single source of truth for "the model to train".
            self.loaded_model_path = model_path
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.prediction_yaml = yaml.safe_load(f)
            
            if 'names' not in self.prediction_yaml:
                raise ValueError("The YAML file does not contain a 'names' section for class names.")

            self.class_names = self.prediction_yaml['names']
            logger.debug(f"Loaded class names: {self.class_names}")

            self.prediction_keypoint_schema = None
            full_schema = self.prediction_yaml.get('keypoint_schema')
            if full_schema:
                self.prediction_keypoint_schema = sanitize_schema(full_schema)
            elif self.prediction_yaml.get('kpt_shape'):
                k = int(self.prediction_yaml['kpt_shape'][0])
                self.prediction_keypoint_schema = sanitize_schema({
                    "names": [f"kp{i}" for i in range(k)],
                    "skeleton": [],
                    'flip_idx': self.prediction_yaml.get('flip_idx'),
                })

            # Verify that the number of classes in the YAML matches the model
            if len(self.class_names) != len(self.model.names):
                mismatch_message = (f"Warning: Number of classes in YAML ({len(self.class_names)}) "
                                    f"does not match the model ({len(self.model.names)}). "
                                    "This may cause issues during prediction.")
                logger.warning(mismatch_message)
                return True, mismatch_message
            
            return True, None
        except Exception as e:
            error_message = f"Error loading model or YAML: {str(e)}"
            logger.exception("Error loading model or YAML")
            return False, error_message
    
    def predict(self, input_data):
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        from ..core.torch_utils import resolve_torch_device
        device, _ = resolve_torch_device()
        if isinstance(input_data, (str, np.ndarray)):
            results = self.model(input_data, conf=self.conf_threshold, save=False, show=False, device=device)
        else:
            raise ValueError("Invalid input type. Expected file path or numpy array.")
        
        # Get the input size used for prediction and the original image size
        input_size = results[0].orig_shape
        original_size = results[0].orig_img.shape[:2]
        return results, input_size, original_size

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf
