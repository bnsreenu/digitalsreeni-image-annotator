"""Model-assisted review: score images by disagreement or uncertainty (#71).

The scoring itself is Qt-free in :mod:`core.disagreement`; this controller runs
the model across the project, feeds it, and turns the result into something the
image list can show and sort by.

Two modes, picked per image rather than by the user:

* an image that **already has annotations** is scored by how much the model
  disagrees with them — a label-error hint;
* an image with **none** is scored by how unsure the model is — an
  annotate-this-next hint.

That split is automatic because it is not a preference: the useful question is
different depending on whether ground truth exists, and asking the user to pick
would just be asking them to restate what the data already says.

Nothing here ever modifies an annotation. Predictions are shown as temp
annotations through the existing review overlay, and the existing accept/reject
path applies.
"""

import os

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..core import disagreement
from ..core.logging_config import get_logger
from ..core.slice_cache import slice_names
from ..core.video_handler import is_video

logger = get_logger(__name__)

MODE_DISAGREEMENT = "disagreement"
MODE_UNCERTAINTY = "uncertainty"


class ReviewController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        # name -> {"score", "mode", "breakdown", "model"}
        self.scores = {}
        # The model identity the current scores belong to. Scores from a
        # previous training round say nothing about the model you just
        # trained, so they are dropped rather than silently reused.
        self.scored_model = None

    # --- collection ---

    def collect_work_items(self):
        """``[(image_name, image_path)]`` for every scorable image.

        Only plain 2D images: prediction runs from a file path, and neither a
        multi-dimensional stack nor a video has one per slice. That limit is
        the same one ``predict_single_image`` already enforces, and it is
        reported rather than silently applied.
        """
        items = []
        for info in self.mw.all_images:
            file_name = info.get("file_name")
            if not file_name:
                continue
            if info.get("is_multi_slice") or info.get("is_video") or is_video(file_name):
                continue
            path = self.mw.image_paths.get(file_name)
            if path and os.path.exists(path):
                items.append((file_name, path))
        return items

    def skipped_count(self):
        """How many images the run cannot cover, for an honest UI message."""
        return len(self.mw.all_images) - len(self.collect_work_items())

    # --- run ---

    def run(self):
        """Score every scorable image with the loaded prediction model."""
        trainer = getattr(self.mw, "yolo_trainer", None)
        if trainer is None or getattr(trainer, "model", None) is None:
            QMessageBox.warning(
                self.mw,
                "Review with model",
                "Load or train a prediction model first.",
            )
            return

        items = self.collect_work_items()
        if not items:
            QMessageBox.information(
                self.mw,
                "Review with model",
                "No plain 2D images to score. Multi-dimensional stacks and "
                "videos are not covered by this run.",
            )
            return

        self.mw.save_current_annotations()

        progress = QProgressDialog(
            "Scoring images…", "Cancel", 0, len(items), self.mw
        )
        progress.setWindowTitle("Review with model")
        progress.setMinimumDuration(0)

        scores = {}
        for index, (file_name, path) in enumerate(items):
            progress.setValue(index)
            progress.setLabelText(f"Scoring {file_name}…")
            QApplication.processEvents()
            if progress.wasCanceled():
                logger.info("Review run cancelled after %d image(s)", index)
                break
            try:
                scores[file_name] = self.score_image(file_name, path, trainer)
            except Exception:
                logger.exception("Scoring failed for %s", file_name)
        progress.setValue(len(items))

        if not scores:
            return

        self.scores = scores
        self.scored_model = getattr(trainer, "model_path", None) or id(trainer.model)
        self.mw.image_controller.refresh_image_list_scores()
        self._report(scores)

    def score_image(self, file_name, path, trainer):
        """Score one image. Returns the score record; never mutates anything."""
        predictions = self.extract_predictions(trainer.predict(path), file_name)

        ground_truth = [
            annotation
            for annotations in (self.mw.all_annotations.get(file_name) or {}).values()
            for annotation in annotations
        ]

        if ground_truth:
            score, breakdown = disagreement.disagreement_score(
                ground_truth, predictions
            )
            mode = MODE_DISAGREEMENT
        else:
            score, breakdown = disagreement.uncertainty_score(predictions)
            mode = MODE_UNCERTAINTY
        return {"score": score, "mode": mode, "breakdown": breakdown}

    def extract_predictions(self, results, file_name):
        """Ultralytics results -> annotation-shaped dicts for scoring.

        Deliberately does **not** go through ``process_yolo_results``: that one
        writes into the review overlay as a side effect, and a scoring run must
        leave the canvas alone. The class names it produces carry the
        ``Temp-`` prefix, which the scorer strips.

        Accepts either the raw results list or the
        ``(results, input_size, original_size)`` triple ``YOLOTrainer.predict``
        actually returns — see :func:`_unwrap_results`.
        """
        predictions = []
        for result in _unwrap_results(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            masks = getattr(result, "masks", None)
            for index, box in enumerate(boxes):
                class_id = int(box.cls)
                entry = {
                    "category_name": names.get(class_id, str(class_id)),
                    "score": float(box.conf),
                }
                if masks is not None and index < len(masks.xy):
                    polygon = masks.xy[index]
                    entry["segmentation"] = [
                        float(c) for point in polygon for c in point
                    ]
                else:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    entry["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                predictions.append(entry)
        return predictions

    def _report(self, scores):
        ranked = disagreement.rank(
            {name: record["score"] for name, record in scores.items()}
        )
        skipped = self.skipped_count()
        top = ", ".join(f"{name} ({value:.1f})" for name, value in ranked[:3])
        message = (
            f"Scored {len(scores)} image(s). Highest disagreement: {top}.\n\n"
            "A high score means the model and the labels differ — it is a hint "
            "worth looking at, not a verdict that the annotation is wrong."
        )
        if skipped:
            message += (
                f"\n\n{skipped} image(s) were not scored: multi-dimensional "
                "stacks and videos are outside this run."
            )
        pose_skipped = sum(
            record["breakdown"].get("skipped_pose", 0) for record in scores.values()
        )
        if pose_skipped:
            message += (
                f"\n\n{pose_skipped} pose instance(s) were excluded: mask IoU is "
                "meaningless for keypoints."
            )
        QMessageBox.information(self.mw, "Review with model", message)

    # --- lookups used by the image list ---

    def score_for(self, file_name):
        record = self.scores.get(file_name)
        return record["score"] if record else None

    def mode_for(self, file_name):
        """Which question this image's score answers, or ``None``.

        Disagreement and uncertainty are not on a common scale, so anything
        ranking several images against each other has to check they are the
        same kind of number first (#82 uses this to decide whether a
        near-duplicate cluster can be ranked by uncertainty at all).
        """
        record = self.scores.get(file_name)
        return record["mode"] if record else None

    def has_scores(self) -> bool:
        return bool(self.scores)

    def clear_scores(self):
        """Drop the scores.

        Called from ``ModelRegistryController.finish_run`` after every
        successful run: a score computed with the previous model says nothing
        about the new one, and a stale ranking painted on the image list is
        worse than none because it still looks authoritative.
        """
        self.scores = {}
        self.scored_model = None
        self.mw.image_controller.refresh_image_list_scores()

    def show_predictions_for_current(self):
        """Put the current image's predictions in the review overlay.

        Goes through ``predict_single_image``, which already routes into
        ``temp_annotations`` — so the disagreement is visible against the
        existing labels and the normal accept/reject path applies. Existing
        annotations are never touched.
        """
        file_name = self.mw.image_file_name
        if not file_name:
            return
        self.mw.predict_single_image(file_name)


def _unwrap_results(returned):
    """The Ultralytics results list out of whatever ``predict`` handed back.

    ``YOLOTrainer.predict`` returns ``(results, input_size, original_size)``,
    not a bare list — ``process_yolo_results`` unpacks it explicitly. Scoring
    did not, and iterating the triple yielded three objects with no ``.boxes``,
    so **every image produced zero predictions**. Nothing raised: annotated
    images simply scored "every label missed" and unannotated ones scored 0,
    which is a ranking that looks entirely plausible and means nothing.

    Normalising here rather than at the one call site is deliberate: this is
    the exact silent failure the module docstring warns about, and a second
    caller passing the triple must not be able to reintroduce it.
    """
    if returned is None:
        return []
    # The triple: first element is the results list, the other two are
    # (height, width) size tuples.
    if (
        isinstance(returned, tuple)
        and len(returned) == 3
        and isinstance(returned[1], tuple)
        and isinstance(returned[2], tuple)
    ):
        return returned[0] or []
    return returned


def unscored_names(slices):
    """Slice names of a stack, for reporting what a run could not cover."""
    return slice_names(slices)
