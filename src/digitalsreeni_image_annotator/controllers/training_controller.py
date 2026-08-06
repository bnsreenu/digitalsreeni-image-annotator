"""One entry point for all training (issue #73, ADR-042).

Dispatches the unified :class:`TrainDialog` to the existing trainers and
performs the mechanics the user never wanted to click through: dataset
preparation, YAML handling, model loading, saving, and selector refresh.

Before this, training a YOLO model *and actually using it* took six menu
navigations and about ten dialogs, including a step in the middle where you
reloaded the model you had just produced.

``YOLOController`` and ``SAMTrainController`` keep their public training
methods — this orchestrates them rather than replacing them, so the GPU gate,
the busy guard, the progress dialog, the stop button and MLflow configuration
all keep working exactly as before. That is deliberate: this is a UI and
orchestration change, not a training-logic change.
"""

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QDialog, QMessageBox

from ..core.logging_config import get_logger
from ..dialogs.train_dialog import TYPE_YOLO, TrainDialog
from .yolo_controller import build_yolo_train_opts

logger = get_logger(__name__)


class TrainingController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def open_dialog(self):
        """Show the training dialog and run whatever it returns."""
        if not hasattr(self.mw, "current_project_dir"):
            QMessageBox.warning(
                self.mw,
                "Train Model",
                "Open or create a project first — training needs somewhere to "
                "write the dataset and the resulting weights.",
            )
            return

        dialog = TrainDialog(self.mw)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        config = dialog.get_config()
        if config["type"] == TYPE_YOLO:
            self.run_yolo(config)
        else:
            self.run_sam(config)

    # --- YOLO ---

    def run_yolo(self, config):
        """Load, prepare and train in one sequence.

        Each of these was a separate menu action with its own OK box. The order
        matters: the base model must be loaded *before* ``train_model`` runs its
        pre-flight comparison of the model's ``.task`` against the prepared
        YAML — the check that turns an opaque Ultralytics failure into an
        actionable message (ADR-029 PR-3). Preserving that check was a stated
        constraint of the issue.
        """
        yolo = self.mw.yolo_controller
        if not self.mw.yolo_trainer:
            yolo.initialize_yolo_trainer()
        trainer = self.mw.yolo_trainer
        if trainer is None:
            return

        # This dialog carries its own split slider, so it never passes through
        # `prompt_validation_split` — the split warning has to be raised here
        # too, or the app's main training path is the one place it is missing
        # (ADR-044). Declining it abandons the run: the user has just been told
        # the validation numbers cannot be trusted.
        #
        # Before `load_model`, which downloads weights on first use: the split
        # is knowable without the model, and backing out should not cost a
        # several-hundred-megabyte download. Only `prepare_dataset` has to come
        # after the load (ADR-042).
        from .io_controller import confirm_split_warning, split_inputs

        names, groups = split_inputs(self.mw)
        if not confirm_split_warning(
            self.mw,
            names,
            self.mw.image_slices,
            config["val_split"],
            groups=groups,
        ):
            return

        # load_model reports its own failure and returns False.
        if not trainer.load_model(config["base_model"]):
            return

        try:
            yaml_path = trainer.prepare_dataset(config["val_split"], groups=groups)
        except Exception as exc:
            logger.exception("Dataset preparation failed")
            QMessageBox.critical(
                self.mw, "Train Model", f"Could not prepare the dataset:\n{exc}"
            )
            return
        logger.info("prepared dataset at %s", yaml_path)

        train_opts = build_yolo_train_opts(
            config["epochs"],
            cos_lr=config["cos_lr"],
            lr0=config["lr0"],
            patience=config["patience"],
        )
        yolo.start_training(config["epochs"], config["imgsz"], train_opts)

    # --- SAM ---

    def run_sam(self, config):
        """Fine-tune SAM 2 on the current project.

        ``SAMTrainController.train_on_project`` was already a genuine one-click
        path — it is the model this issue follows, not something to replace.
        The GPU gate and the busy guard live inside it and still apply.
        """
        self.mw.sam_train_controller.train_on_project()
