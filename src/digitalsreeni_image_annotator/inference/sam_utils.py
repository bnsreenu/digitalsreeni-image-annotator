"""
SAM 2 utilities — runs Ultralytics SAM in-process.

History
-------
Earlier versions delegated to a ``sam_worker.py`` subprocess to dodge
``WinError 1114`` on Windows + Python 3.14 + PyQt5 (the now-superseded
ADR-011). Migrating to PyQt6 (ADR-014) eliminated that DLL load-order
conflict, so we run the model directly here — saves a ~1-2 s spawn
per call and lets us keep the model resident. See ADR-013 for the
threading and re-entrancy story.

Threading model
---------------
Inference runs on a worker thread (QThread) so the UI stays
responsive. The public API still looks synchronous — the caller
gets the result returned — but the call site's thread (typically
the UI thread) keeps pumping events via a nested QEventLoop while
the worker churns. The Qt event loop processing during the wait
means button clicks, redraws and progress dialog cancels all
continue to flow. Callers that disabled buttons before the call
remain protected from re-entry; callers that didn't (e.g. simple
click-segment) should make sure they themselves are idempotent
under a possible second click.

torch / ultralytics are imported lazily on first inference so app
startup stays fast for users who never touch SAM.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import QEventLoop, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ..utils import models_base_dir

from ..core.logging_config import get_logger

logger = get_logger(__name__)


MODEL_NAMES = [
    "SAM 2 tiny",
    "SAM 2 small",
    "SAM 2 base",
    "SAM 2 large",
    "SAM 2.1 tiny",
    "SAM 2.1 small",
    "SAM 2.1 base",
    "SAM 2.1 large",
]

MODEL_FILES = {
    "SAM 2 tiny": "sam2_t.pt",
    "SAM 2 small": "sam2_s.pt",
    "SAM 2 base": "sam2_b.pt",
    "SAM 2 large": "sam2_l.pt",
    "SAM 2.1 tiny": "sam2.1_t.pt",
    "SAM 2.1 small": "sam2.1_s.pt",
    "SAM 2.1 base": "sam2.1_b.pt",
    "SAM 2.1 large": "sam2.1_l.pt",
}

# SAM weights live under <models_base>/sam/, parallel to DINO models.
SAM_MODELS_DIR = os.path.join(models_base_dir(), "sam")


def _qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """QImage → RGB numpy array. Returned array is a fresh copy.

    The naive ``np.frombuffer(qimage.constBits().asarray(N))`` aliases
    the QImage's pixel buffer. That's a problem in two ways: (1) the
    returned array is invalidated if the QImage is mutated or freed,
    and (2) we hand the array across a thread boundary to the
    inference worker, where Qt's threading rules make any read from
    the QImage memory dicey. Always ``.copy()`` so the worker thread
    owns its own buffer for the duration of the call.
    """
    width = qimage.width()
    height = qimage.height()
    fmt = qimage.format()

    if fmt == QImage.Format.Format_Grayscale8:
        buffer = qimage.constBits().asarray(height * width)
        img = np.frombuffer(buffer, np.uint8).reshape((height, width))
        return np.stack((img,) * 3, -1)  # np.stack already returns a copy

    if fmt in (
        QImage.Format.Format_RGB32,
        QImage.Format.Format_ARGB32,
        QImage.Format.Format_ARGB32_Premultiplied,
    ):
        buffer = qimage.constBits().asarray(height * width * 4)
        img = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
        return img[:, :, :3].copy()

    if fmt == QImage.Format.Format_RGB888:
        buffer = qimage.constBits().asarray(height * width * 3)
        img = np.frombuffer(buffer, np.uint8).reshape((height, width, 3))
        return img.copy()

    # Fallback: convert via Qt. ``converted`` is a local QImage that
    # goes out of scope at function return, so we MUST copy before
    # the buffer is freed.
    converted = qimage.convertToFormat(QImage.Format.Format_RGB32)
    buffer = converted.constBits().asarray(height * width * 4)
    img = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    return img[:, :, :3].copy()


# ── geometry helpers ────────────────────────────────────────────────────────

def _mask_to_polygon(mask: np.ndarray) -> list | None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    if not polygons:
        return None
    biggest = max(
        polygons,
        key=lambda p: cv2.contourArea(np.array(p).reshape(-1, 2)),
    )
    return biggest


def _bbox_of_contour(contour: list) -> tuple[float, float, float, float]:
    pts = np.array(contour).reshape(-1, 2)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _bbox_area(bbox: list) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _check_points(contour: list, positive: list, negative: list) -> bool:
    cnt = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    for x, y in positive:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) < 0:
            return False
    for x, y in negative:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
            return False
    return True


def _predicted_bbox_area_ratio(pred_contour: list, user_bbox: list) -> float:
    px1, py1, px2, py2 = _bbox_of_contour(pred_contour)
    user_area = _bbox_area(user_bbox)
    if user_area == 0:
        return 0.0
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    return pred_area / user_area


def _bbox_constraints_ok(contour, user_bbox) -> bool:
    ratio = _predicted_bbox_area_ratio(contour, user_bbox)
    if ratio < 0.20:
        return False
    ux, uy, ux2, uy2 = user_bbox
    uw, uh = ux2 - ux, uy2 - uy
    px, py, px2, py2 = _bbox_of_contour(contour)
    pw, ph = px2 - px, py2 - py
    if pw < 0.5 * uw or ph < 0.5 * uh:
        return False
    if pw > 1.5 * uw or ph > 1.5 * uh:
        return False
    return True


# ── threading scaffolding ──────────────────────────────────────────────────

class _InferenceThread(QThread):
    """Runs a callable on a background thread.

    Captures both the return value AND any exception raised, so
    ``_run_sync`` can re-raise on the calling thread. Swallowing
    exceptions inside the worker was the cause of silent
    model-load failures (review P0).

    We use QThread (not QRunnable) because QRunnable's signal/slot
    story requires a separate QObject anyway and we want a minimal
    wrapper. Lifetime is bounded by the QEventLoop in _run_sync.
    """

    finished_with_result = pyqtSignal()

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._result = None
        self._exc: BaseException | None = None

    def run(self):
        try:
            self._result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:  # noqa: BLE001 - rebroadcast verbatim
            # Capture rather than print — _run_sync will re-raise on the
            # calling thread so try/except at the call site actually catches.
            # We DO emit a traceback to the console here so the failure
            # mode is visible even if a caller swallows the exception in
            # a broad except block (debugging silent-failure regressions).
            logger.exception("Exception in inference thread")
            self._exc = exc
        self.finished_with_result.emit()


class InferenceBusyError(RuntimeError):
    """Raised when ``_run_sync`` is re-entered before the first call returns.

    See ``_run_sync`` for the full story. Callers that drive inference
    from timers or user events should catch this and skip rather than
    treating it as "no result found".
    """


# Module-level busy flag. ``_run_sync`` pumps the calling thread's
# event loop while inference runs, so a timer fire or user click can
# call back into ``_run_sync`` on the same thread before the first
# call returns. Two concurrent ``model(...)`` calls would race on the
# torch/ultralytics object (not thread-safe) and produce garbled
# masks or CUDA errors. A QMutex won't help: it's the same thread
# trying to re-acquire, which deadlocks a non-recursive mutex and is
# meaningless for a recursive one. A simple flag with an explicit
# exception is the honest fix — callers learn about the re-entry
# instead of silently getting None back.
_inference_in_flight = False


def _run_sync(fn, *args, **kwargs):
    """Run fn on a worker thread; pump the calling thread's event loop
    until done; return the result. Re-raises exceptions on the caller.

    Looks synchronous to callers but keeps the UI alive — timers,
    repaints and progress dialog cancels continue to fire during the
    wait. Re-entry from the same thread (the only kind that can happen
    here) raises :class:`InferenceBusyError` rather than corrupting
    the model with concurrent forward passes.

    **Call from the GUI thread only.** The module-level
    ``_inference_in_flight`` flag is not protected for cross-thread
    access; if a future contributor drives inference from a non-GUI
    worker thread (e.g. a background patching/training thread), the
    flag becomes a true race. The check below is a tripwire — kept
    as an explicit ``raise`` rather than ``assert`` so it survives
    ``python -O``.
    """
    from PyQt6.QtCore import QCoreApplication, QThread as _QThread
    app = QCoreApplication.instance()
    if app is not None and _QThread.currentThread() is not app.thread():
        raise RuntimeError(
            "_run_sync must be called from the GUI thread. "
            "See ADR-013 — the re-entry guard is GUI-thread-local."
        )
    global _inference_in_flight
    if _inference_in_flight:
        raise InferenceBusyError(
            "Another SAM/DINO inference is still running. "
            "Wait for it to finish or cancel before issuing a new call."
        )
    _inference_in_flight = True
    try:
        thread = _InferenceThread(fn, *args, **kwargs)
        loop = QEventLoop()
        thread.finished_with_result.connect(loop.quit)
        thread.start()
        loop.exec()
        thread.wait()
        if thread._exc is not None:
            raise thread._exc
        return thread._result
    finally:
        _inference_in_flight = False


# ── public class ───────────────────────────────────────────────────────────

class SAMUtils(QObject):
    """Runs Ultralytics SAM 2 in-process with a cached model."""

    # Exposed for backward compat with annotator_window.py UI setup
    sam_models = MODEL_FILES.copy()

    model_changed = pyqtSignal(str)  # emitted with new model name after load

    def __init__(self):
        super().__init__()
        self.current_sam_model: str | None = None
        self._model = None  # ultralytics.SAM instance once loaded
        self._loaded_model_file: str | None = None
        self._device: str | None = None  # resolved at model load
        # Fine-tuned checkpoints: {display_name: checkpoint_path}. Populated
        # from training.sam_trainer.list_custom_models() so they load through
        # the same SAM(path) path as the built-ins (see SAM fine-tuning ADR).
        self.custom_models: dict[str, str] = {}

    def register_custom_models(self, mapping: dict) -> None:
        """Merge fine-tuned model entries so they become selectable."""
        self.custom_models.update(mapping or {})

    def _resolve_model_file(self, model_name: str) -> str:
        if model_name in MODEL_NAMES:
            return os.path.join(SAM_MODELS_DIR, MODEL_FILES[model_name])
        if model_name in self.custom_models:
            return self.custom_models[model_name]
        raise ValueError(f"Unknown SAM model: {model_name}")

    # ── model lifecycle ────────────────────────────────────────────────

    def change_sam_model(self, model_name: str) -> None:
        if model_name == "Pick a SAM Model":
            self.current_sam_model = None
            self._model = None
            self._loaded_model_file = None
            logger.info("SAM model unset")
            return

        # Resolve to a checkpoint path first (built-in name or fine-tuned
        # model) so an unknown name fails before we touch a worker thread.
        model_file = self._resolve_model_file(model_name)

        # Load on a worker thread to avoid stalling the UI on the
        # ~1-3 s torch model-load. Behaves synchronously to callers and
        # re-raises any load-time exception (network, corrupt weights,
        # CUDA OOM) — only flip `current_sam_model` AFTER success so
        # callers don't see a stale name on failure.
        _run_sync(self._load_model_blocking, model_file)
        self.current_sam_model = model_name
        self.model_changed.emit(model_name)
        logger.info(f"SAM model loaded: {model_name}")

    def _load_model_blocking(self, model_file: str) -> None:
        # Lazy import keeps app startup fast for users who never use SAM.
        from ultralytics import SAM
        from ..core.torch_utils import resolve_torch_device

        self._device, _ = resolve_torch_device()
        self._log_device()
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        self._model = SAM(model_file)
        self._loaded_model_file = model_file

    def _log_device(self) -> None:
        try:
            import torch
            if self._device == "cuda":
                dev = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA: {torch.version.cuda} — {dev}")
            else:
                logger.info("Running on CPU")
        except Exception:
            logger.debug("could not query torch device", exc_info=True)

    def unload(self) -> None:
        """Free GPU/CPU memory held by the loaded model.

        Three-step recipe required to actually drop the GPU allocation:

        1. Move the underlying nn.Module to CPU
           (``self._model.model.cpu()``). Without this the GPU tensors
           stay resident even after Python drops its reference, because
           the allocator pool keeps them.
        2. Drop the Python references, ``gc.collect()`` to break the
           Ultralytics circular refs.
        3. ``torch.cuda.empty_cache()`` + ``ipc_collect()`` +
           ``synchronize()`` so the allocator releases freed blocks
           back to the driver and pending kernels complete first.

        Caveat: PyTorch keeps a CUDA context per process. Even after a
        clean unload, Task Manager / ``nvidia-smi`` show the residual
        context (~200-500 MB depending on driver). To reclaim that the
        user must restart the app.
        """
        import gc
        try:
            if self._model is not None and hasattr(self._model, "model"):
                self._model.model.cpu()
        except Exception:
            logger.warning(
                "unload: moving model to CPU failed; GPU memory may not be "
                "fully released", exc_info=True,
            )
        self._model = None
        self.current_sam_model = None
        self._loaded_model_file = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.warning(
                "CUDA cache cleanup failed during unload; GPU memory may not "
                "be fully released", exc_info=True,
            )
        logger.info("unload complete")

    # ── inference ──────────────────────────────────────────────────────

    def apply_sam_points(self, image: QImage, positive_points, negative_points):
        if not self.current_sam_model or self._model is None:
            logger.warning("No SAM model selected.")
            return None
        if not positive_points:
            logger.warning("No positive points for SAM-points")
            return None
        return _run_sync(
            self._sam_points_blocking,
            _qimage_to_numpy(image),
            list(positive_points),
            list(negative_points),
        )

    def _sam_points_blocking(self, image_np, positive_points, negative_points):
        all_points = [positive_points + negative_points]
        all_labels = [([1] * len(positive_points)) + ([0] * len(negative_points))]
        results = self._model(
            image_np, points=all_points, labels=all_labels, device=self._device
        )

        masks = results[0].masks.data.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        best_result = None
        best_score = -1.0
        for i, mask in enumerate(masks):
            contour = _mask_to_polygon(mask)
            if contour is None:
                continue
            if not _check_points(contour, positive_points, negative_points):
                continue
            mask_pixels = int(mask.sum())
            if mask_pixels > best_score:
                score = float(confidences[i]) if i < len(confidences) else 0.0
                best_score = mask_pixels
                best_result = {"segmentation": contour, "score": score}

        return best_result

    def apply_sam_prediction(self, image: QImage, bbox):
        if not self.current_sam_model or self._model is None:
            logger.warning("No SAM model selected.")
            return None
        return _run_sync(
            self._sam_bbox_blocking,
            _qimage_to_numpy(image),
            list(bbox),
        )

    def _sam_bbox_blocking(self, image_np, bbox):
        results = self._model(image_np, bboxes=[bbox], device=self._device)
        res = results[0]
        if not (hasattr(res, "masks") and res.masks is not None):
            return None

        masks = res.masks.data.cpu().numpy()
        confidences = (
            res.boxes.conf.cpu().numpy()
            if hasattr(res.boxes, "conf")
            else np.zeros(len(masks))
        )

        best = None
        best_pixels = -1
        for i, mask in enumerate(masks):
            contour = _mask_to_polygon(mask)
            if contour is None:
                continue
            if not _bbox_constraints_ok(contour, bbox):
                continue
            pixels = int(mask.sum())
            if pixels > best_pixels:
                best_pixels = pixels
                score = float(confidences[i]) if i < len(confidences) else 0.0
                best = {"segmentation": contour, "score": score}

        return best

    def apply_sam_predictions_batch(self, image: QImage, bboxes: list):
        if not self.current_sam_model or self._model is None:
            logger.warning("apply_sam_predictions_batch: no SAM model selected")
            return None
        if not bboxes:
            logger.debug("apply_sam_predictions_batch: empty bbox list")
            return []
        logger.debug(f"apply_sam_predictions_batch: running on {len(bboxes)} bbox(es)")
        return _run_sync(
            self._sam_batch_blocking,
            _qimage_to_numpy(image),
            [list(b) for b in bboxes],
        )

    def _sam_batch_blocking(self, image_np, bboxes):
        results = self._model(image_np, bboxes=bboxes, device=self._device)
        res = results[0]
        if not (hasattr(res, "masks") and res.masks is not None):
            # Build a fresh dict per bbox so callers can mutate one
            # entry without affecting the others (a `[d] * N` would
            # alias the same dict N times).
            return [{"error": "No mask generated."} for _ in bboxes]

        masks = res.masks.data.cpu().numpy()
        confidences = (
            res.boxes.conf.cpu().numpy()
            if hasattr(res.boxes, "conf")
            else np.zeros(len(masks))
        )

        output = []
        for i in range(len(masks)):
            mask = masks[i]
            score = float(confidences[i]) if i < len(confidences) else 0.0
            contour = _mask_to_polygon(mask)
            if contour is None:
                output.append({"error": "No valid mask polygon."})
                continue

            user_bbox = bboxes[i]
            if not _bbox_constraints_ok(contour, user_bbox):
                output.append({"error": "Mask failed bbox constraints."})
                continue

            output.append({"segmentation": contour, "score": score})
        return output
