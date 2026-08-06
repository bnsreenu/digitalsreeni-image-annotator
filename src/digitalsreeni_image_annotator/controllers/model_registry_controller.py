"""Post-training lifecycle: register, save, report, try (issue #74).

Training used to finish with a message box saying `Training complete`. Then:
the model you had just trained was **not** selected for prediction, so you
navigated to Prediction Settings → Load Model and loaded it by hand; if you had
not separately invoked Save Model it lived wherever Ultralytics put it; you
were told nothing about whether it was any good; and to find out you set up a
prediction run yourself.

The manual reload in the middle was the sharpest edge in the whole workflow.

Both trainers converge here so YOLO and SAM behave identically at the end of a
run. Registration reuses ``_register_trained_model`` exactly — the two-tier
keypoint-schema logic (rich embedded schema when every trained class shares
one, else generic names reconstructed from bare ``kpt_shape``/``flip_idx``) is
subtle and must not be reimplemented here (ADR-029 PR-3).
"""

import os
import shutil
from datetime import datetime

from PyQt6.QtCore import QObject

from ..core import model_sidecar
from ..core.logging_config import get_logger

logger = get_logger(__name__)

MODELS_SUBDIR = "models"


class ModelRegistryController(QObject):
    """Runs the four post-training steps and remembers the last run."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.last_run = None

    # --- entry point ---

    def finish_run(self, *, model_type, result, weights_path, metrics=None,
                   config=None, mlflow_url=None):
        """Handle a completed run and return a summary dict for the panel.

        Returns ``None`` for a failed or stopped run — nothing is registered and
        nothing is written. Both trainers hand back a result that can represent
        an error, and registering on one would put a broken model in front of
        the user as though it were ready.
        """
        if isinstance(result, str):
            logger.info("run failed; nothing registered: %s", result)
            return None
        if not weights_path or not os.path.exists(weights_path):
            logger.info("run produced no weights at %s; nothing registered",
                        weights_path)
            return None
        # Autosave and project writes are suspended during a load (ADR-005);
        # writing a model into the project mid-load would be the same hazard.
        if getattr(self.mw, "is_loading_project", False):
            logger.warning("project is loading; skipping post-training save")
            return None

        saved_path, sidecar = self.save_into_project(
            model_type=model_type,
            weights_path=weights_path,
            metrics=metrics,
            config=config,
        )
        summary = {
            "model_type": model_type,
            "weights_path": saved_path or weights_path,
            "sidecar_path": sidecar,
            "metrics": metrics or {},
            "config": config or {},
            "mlflow_url": mlflow_url,
        }
        self.last_run = summary
        # Review scores were computed with the PREVIOUS model, so they say
        # nothing about this one. Dropping them beats leaving a stale ranking
        # painted on the image list (issue #71).
        review = getattr(self.mw, "review_controller", None)
        if review is not None:
            review.clear_scores()
        return summary

    # --- (b) save into the project ---

    def models_dir(self):
        """``<project>/models``, or None when there is no project directory."""
        project_dir = getattr(self.mw, "current_project_dir", None)
        return os.path.join(project_dir, MODELS_SUBDIR) if project_dir else None

    def save_into_project(self, *, model_type, weights_path, metrics=None,
                          config=None):
        """Copy the weights under ``<project>/models`` with a JSON sidecar.

        Returns ``(saved_path, sidecar_path)``; either may be ``None`` if the
        step could not run. No file dialog: where the weights go is not a
        decision worth interrupting a run for.

        Every checkpoint is kept. Pruning to the last N was considered and
        rejected — deleting a model the user has not copied anywhere is not
        recoverable, and the results panel reports the directory size instead so
        the growth is visible rather than silently managed.
        """
        directory = self.models_dir()
        if not directory:
            logger.info("no project directory; leaving weights at %s", weights_path)
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = os.path.splitext(os.path.basename(weights_path))[0]
        if base_name in ("best", "last"):
            # Ultralytics names every run's output best.pt; that tells the user
            # nothing once several runs sit in one directory.
            base_name = f"{model_type}_{base_name}"
        target = model_sidecar.unique_weights_path(directory, base_name, timestamp)

        try:
            shutil.copy2(weights_path, target)
        except OSError:
            logger.exception("could not copy the trained weights into the project")
            return None, None

        payload = model_sidecar.build_sidecar(
            model_type=model_type,
            task=self._task_of(model_type),
            class_names=self._class_names(),
            keypoint_schema=self._shared_keypoint_schema(),
            config=config,
            metrics=metrics,
            timestamp=timestamp,
        )
        try:
            sidecar = model_sidecar.write_sidecar(target, payload)
        except OSError:
            logger.exception("could not write the model sidecar")
            sidecar = None
        logger.info("saved trained model to %s", target)
        return target, sidecar

    def _task_of(self, model_type):
        if model_type != "yolo":
            return None
        trainer = getattr(self.mw, "yolo_trainer", None)
        model = getattr(trainer, "model", None)
        return getattr(model, "task", None)

    def _class_names(self):
        return [
            name
            for name in getattr(self.mw, "class_mapping", {})
            if not name.startswith("Temp-")
        ]

    def _shared_keypoint_schema(self):
        """The keypoint schema iff every trained class shares one.

        Same two-tier rule ``_register_trained_model`` applies: a rich embedded
        schema only when it is unambiguous, otherwise nothing and let the
        bare-``kpt_shape`` reconstruction handle it. Recording one class's
        schema for a mixed project would be worse than recording none.
        """
        schemas = getattr(self.mw, "keypoint_schemas", {}) or {}
        relevant = [schemas.get(name) for name in self._class_names()]
        relevant = [schema for schema in relevant if schema]
        if not relevant or len(relevant) != len(self._class_names()):
            return None
        first = relevant[0]
        return first if all(schema == first for schema in relevant) else None

    def models_dir_size_mb(self):
        """Total size of ``<project>/models`` in MB, or None.

        Surfaced in the results panel: every run leaves a checkpoint, and the
        honest answer to that is to show the number rather than quietly delete
        old ones.
        """
        directory = self.models_dir()
        if not directory or not os.path.isdir(directory):
            return None
        total = 0
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024 * 1024)

    # --- (d) try it now ---

    def can_try_now(self):
        """True when a one-click try is meaningful.

        **YOLO only.** ``predict_single_image`` routes to the YOLO trainer
        regardless of what was trained, so offering it after a SAM fine-tune
        would run the loaded YOLO model — or pop "No Model" — while the panel
        claimed the SAM model was active. A fine-tuned SAM checkpoint is used
        interactively via SAM-box / SAM-points, and
        ``SAMTrainController.training_finished`` already selects it in the SAM
        dropdown.
        """
        if (self.last_run or {}).get("model_type") != "yolo":
            return False
        return bool(getattr(self.mw, "image_file_name", ""))

    def try_on_current_image(self):
        """Run the fresh model on the current image into the review overlay.

        Goes through ``predict_single_image``, which already routes predictions
        into ``temp_annotations`` — closing the loop from train to see to fix
        without a second review mechanic.
        """
        if not self.can_try_now():
            return
        self.mw.predict_single_image(self.mw.image_file_name)
