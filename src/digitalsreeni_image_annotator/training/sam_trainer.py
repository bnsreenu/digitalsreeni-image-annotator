"""SAM 2 / 2.1 fine-tuning engine — Ultralytics-native custom training loop.

Why this exists
---------------
Ultralytics has **no SAM trainer**: ``SAM.task_map`` registers only a
*predictor* for the ``segment`` task, so ``SAM(...).train()`` raises
``NotImplementedError`` (verified on ultralytics 8.4.51). We therefore
cannot mirror the YOLO ``model.train(data=yaml, ...)`` path.

What we do instead
------------------
``SAM(...).model`` is a plain ``nn.Module`` (``SAM2Model``) exposing
``image_encoder`` / ``sam_prompt_encoder`` / ``sam_mask_decoder``, and the
Ultralytics ``SAM2Predictor`` already implements the forward path in reusable
pieces — ``get_im_features`` (image encoder) and ``prompt_inference`` /
``_inference_features`` (prompt encoder + mask decoder). We call those methods
**under autograd** (they are not wrapped in ``inference_mode`` unless reached
via the public ``__call__``), add a focal+dice loss and an AdamW step, and
save a checkpoint that reloads through the existing ``SAM(path)`` inference
path. No extra dependency, and the result drops straight into the app's SAM
model selector.

This keeps our exposure confined to a thin adapter over already-exercised
predictor methods. The spike that validated the mechanic lives in the PR for
issue bnsreenu#73.

Threading
---------
``train()`` is **blocking and CPU/GPU-bound**; the controller runs it on a
dedicated ``QThread`` (never the GUI thread). Unlike SAM inference it does
*not* go through ``sam_utils._run_sync`` — that helper's re-entry guard is
GUI-thread-local.

The trainer loads its **own** ``SAM`` instance (see ``_build_predictor``); it
never touches ``SAMUtils._model``, so this is not "two threads driving one
model". The hazard it *does* create is two SAM models (the resident inference
one and this training one) competing for the same GPU/CUDA context. The
controller therefore locks the SAM inference UI (tools + model selector + the
fine-tune menu) for the duration so no concurrent inference or model swap can
be triggered while a run is in flight.
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ..inference.sam_utils import MODEL_FILES, MODEL_NAMES, SAM_MODELS_DIR

# Fine-tuned checkpoints live alongside the base weights, namespaced so they
# never collide with Ultralytics' auto-downloaded base files.
SAM_CUSTOM_DIR = os.path.join(SAM_MODELS_DIR, "custom")


def make_custom_filename(base_model: str, name: str) -> str:
    """Build a fine-tuned checkpoint path under ``SAM_CUSTOM_DIR``.

    Ultralytics' ``build_sam`` selects the architecture by ``ckpt.endswith(token)``
    where ``token`` is the base file name (e.g. ``sam2_t.pt``). A fine-tuned file
    therefore **must** keep that suffix or ``SAM(path)`` raises "not a supported
    SAM model". We sanitise the user label and append ``_<base_token>``.
    """
    token = MODEL_FILES.get(base_model, os.path.basename(base_model))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    safe = safe or "finetuned"
    return os.path.join(SAM_CUSTOM_DIR, f"{safe}_{token}")


def list_custom_models() -> dict:
    """``{display_name: path}`` for fine-tuned checkpoints, for the SAM selector."""
    out = {}
    if os.path.isdir(SAM_CUSTOM_DIR):
        for fn in sorted(os.listdir(SAM_CUSTOM_DIR)):
            if fn.endswith(".pt"):
                out[f"★ {os.path.splitext(fn)[0]}"] = os.path.join(SAM_CUSTOM_DIR, fn)
    return out


# ── geometry: annotation → (mask, prompt) ───────────────────────────────────

def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Flat ``[x1,y1,x2,y2,...]`` polygon → bool mask (inverse of
    ``sam_utils._mask_to_polygon``)."""
    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def bbox_to_mask(bbox, height: int, width: int) -> np.ndarray:
    """``[x, y, w, h]`` → bool mask."""
    x, y, w, h = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 1, thickness=-1)
    return mask.astype(bool)


def mask_to_xyxy(mask: np.ndarray):
    """Tight ``[x1, y1, x2, y2]`` bounding box of a bool mask, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_to_point(mask: np.ndarray):
    """A single foreground point well inside the mask.

    Uses the distance transform's argmax so the point sits near the medial
    axis rather than on an edge — a more stable positive prompt than a random
    interior pixel.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [float(x), float(y)]


# ── loss ────────────────────────────────────────────────────────────────────

def _focal_dice_loss(logits, target, focal_weight: float = 20.0):
    """SAM's mask supervision: focal + dice, ≈20:1 (focal:dice).

    ``logits`` and ``target`` are ``(1, 1, H, W)`` on the same device; target
    is float {0,1}. Matches the recipe used across the SAM fine-tuning
    literature (focal for hard pixels, dice for region overlap).
    """
    import torch
    import torch.nn.functional as F

    prob = torch.sigmoid(logits)
    # Focal (binary), gamma=2.
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    focal = (bce * (1 - p_t).pow(2)).mean()
    # Dice.
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return focal_weight * focal + dice


# ── dataset ──────────────────────────────────────────────────────────────────

class SampleGroup:
    """One image plus the specs for its instances, loaded lazily.

    ``image_loader`` returns an RGB ``uint8`` array (matching what
    ``sam_utils._qimage_to_numpy`` feeds inference — channel-order consistency
    between train and predict matters more than absolute order). ``specs`` are
    raw annotations (``{"segmentation": [...]}`` or ``{"bbox": [x,y,w,h]}``)
    rasterised to bool masks **at load time** using the actual image size, so
    masks are never held in RAM between epochs and always match the image.
    """

    def __init__(self, image_loader, specs, name: str = ""):
        self._image_loader = image_loader
        self.specs = specs
        self.n_instances = len(specs)
        # Source image/slice name — used only to key the deterministic train/val
        # split (see sam_dataset.split_groups); loading never touches it.
        self.name = name

    def load(self):
        """Return ``(image_rgb, [{"mask": bool HxW}, ...])``."""
        image = self._image_loader()
        h, w = image.shape[:2]
        instances = []
        for spec in self.specs:
            if spec.get("segmentation"):
                mask = polygon_to_mask(spec["segmentation"], h, w)
            elif spec.get("bbox"):
                mask = bbox_to_mask(spec["bbox"], h, w)
            else:
                continue
            if mask.any():
                instances.append({"mask": mask})
        return image, instances


# ── engine ───────────────────────────────────────────────────────────────────

class SAMFineTuner(QObject):
    """Fine-tunes a SAM 2 mask decoder (optionally image encoder) on
    user instances. Mirrors ``YOLOTrainer``'s signal/stop surface so the
    controller and progress dialog wiring is identical."""

    progress_signal = pyqtSignal(str)
    # Emitted (from the worker thread) with the MLflow-UI deep link once the run
    # opens, so the controller can show a clickable link + auto-open the browser.
    mlflow_run_url = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_training = False

    def stop_training_signal(self):
        self.stop_training = True
        self.progress_signal.emit("Stopping after current step…")

    # -- model setup ---------------------------------------------------------

    def _build_predictor(self, base_model):
        """Return a ready ``SAM2Predictor`` for ``base_model`` (a registry name
        like ``"SAM 2 tiny"`` or a path to a ``.pt``).

        Forces predictor creation with one throwaway predict so ``set_image`` /
        ``prompt_inference`` are usable. Pins the device via
        ``resolve_torch_device`` so an incompatible GPU (which Ultralytics would
        otherwise pick blindly and crash on) is honoured as CPU — the same
        device decision SAM/DINO/YOLO inference already share."""
        from ultralytics import SAM

        from ..core.torch_utils import resolve_torch_device

        if base_model in MODEL_NAMES:
            model_file = os.path.join(SAM_MODELS_DIR, MODEL_FILES[base_model])
        else:
            model_file = base_model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Base SAM weights not found: {model_file}")

        device, _ = resolve_torch_device()
        model = SAM(model_file)
        warm = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        # device= forces the warmup (and predictor model placement) onto the
        # resolved device rather than Ultralytics' default cuda-if-present.
        model(warm, bboxes=[[8, 8, 56, 56]], device=device, verbose=False)
        return model, model.predictor, model_file

    @staticmethod
    def _device_label(device) -> str:
        """``cuda:0 (NVIDIA GeForce RTX 4070)`` — make it obvious the GPU is in
        use (a bare ``cuda:0`` reads to users like the GPU wasn't detected)."""
        try:
            import torch
            if str(device).startswith("cuda"):
                idx = device.index if getattr(device, "index", None) is not None else 0
                return f"{device} ({torch.cuda.get_device_name(idx)})"
        except Exception:
            pass
        return str(device)

    @staticmethod
    def _apply_freeze(net, freeze_image_encoder: bool):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        for p in net.sam_mask_decoder.parameters():
            p.requires_grad_(True)
        if not freeze_image_encoder:
            for p in net.image_encoder.parameters():
                p.requires_grad_(True)
            net.image_encoder.train()
        trainable = [p for p in net.parameters() if p.requires_grad]
        n = sum(p.numel() for p in trainable)
        return trainable, n

    # -- training ------------------------------------------------------------

    def train(
        self,
        base_model,
        groups,
        *,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 1,
        freeze_image_encoder: bool = True,
        prompt_type: str = "bbox",
        train_pct: float = 80.0,
        patience: int = 20,
        use_lr_schedule: bool = True,
        seed: int = 0,
        out_path: str,
        tracker=None,
    ) -> dict:
        """Fine-tune and save. Returns a small result dict; raises on failure.

        ``groups`` is an iterable of :class:`SampleGroup`. ``batch_size`` is a
        gradient-accumulation count over **images** — all of an image's objects
        are backpropagated together (one backward per image), so the optimizer
        steps every ``batch_size`` images.

        ``train_pct`` holds the rest out as a deterministic per-image validation
        set (issue bnsreenu#85): each epoch logs ``val_loss`` alongside
        ``train_loss``, ``patience`` enables early stopping on ``val_loss`` (0
        disables), and the **best-val** checkpoint is saved rather than the last
        epoch's. At 100% train (or a single image) there is no val set, so the
        val pass / early stopping are skipped and the final epoch is saved.
        ``lr`` is the **peak** LR; with ``use_lr_schedule`` it ramps up over the
        first 10% of optimizer steps then cosine-decays to a 10% floor (else it
        stays constant). ``seed`` makes the split and per-epoch shuffle
        reproducible.

        ``tracker`` is a :class:`~..training.mlflow_tracker.MLflowTracker`.
        The MLflow run is opened, logged to and closed *here* (on the worker
        thread) because MLflow runs are thread-bound. The GUI always supplies
        one; when ``tracker`` is None (direct/programmatic calls, tests) a
        ``_NullTracker`` no-op stands in so all tracking calls are safe.
        """
        import torch
        from ultralytics.utils import ops

        from .mlflow_tracker import _NullTracker
        if tracker is None:
            tracker = _NullTracker()
        # Route tracker status lines through the thread-safe progress signal.
        tracker.set_log(self.progress_signal.emit)
        # And the run's UI deep link through its own signal (GUI handles it).
        tracker.set_run_url_callback(self.mlflow_run_url.emit)

        groups = list(groups)
        if not groups:
            raise ValueError("No annotated instances to train on.")
        if prompt_type not in ("bbox", "point"):
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        # Deterministic per-image hold-out (issue bnsreenu#85). At 100% train (or
        # a single image) val is empty and the val pass / early stopping are off.
        from .sam_dataset import split_groups
        train_groups, val_groups = split_groups(groups, train_pct)

        model, pred, base_file = self._build_predictor(base_model)
        net = pred.model
        device = next(net.parameters()).device
        trainable, n_trainable = self._apply_freeze(net, freeze_image_encoder)
        self.progress_signal.emit(
            f"Base: {os.path.basename(base_file)} | device: {self._device_label(device)} | "
            f"images: {len(train_groups)} train / {len(val_groups)} val | "
            f"trainable params: {n_trainable:,} | "
            f"encoder {'TRAINED' if not freeze_image_encoder else 'frozen'}"
        )
        if not val_groups:
            self.progress_signal.emit(
                "No validation set (train 100% or single image): val_loss and "
                "early stopping are off; saving the final epoch."
            )

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.1)
        # The LR schedule steps once per optimizer step. With gradient
        # accumulation that's ~ceil(train_images / batch_size) steps per epoch;
        # the cosine lambda clamps if the real count drifts (images with no
        # usable instance are skipped), so the estimate is fine.
        scheduler = None
        if use_lr_schedule:
            from torch.optim.lr_scheduler import LambdaLR

            from .lr_schedule import warmup_cosine_lambda
            steps_per_epoch = max(1, (len(train_groups) + batch_size - 1) // batch_size)
            scheduler = LambdaLR(optimizer, warmup_cosine_lambda(epochs * steps_per_epoch))

        self.stop_training = False
        total_instances = sum(g.n_instances for g in groups)
        if total_instances == 0:
            raise ValueError("No annotated instances to train on.")

        tracker.start({
            "base_model": base_model,
            "base_file": os.path.basename(base_file),
            "device": self._device_label(device),
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "freeze_image_encoder": freeze_image_encoder,
            "prompt_type": prompt_type,
            "train_pct": train_pct,
            "patience": patience,
            "lr_schedule": "warmup_cosine" if use_lr_schedule else "constant",
            "images": len(groups),
            "train_images": len(train_groups),
            "val_images": len(val_groups),
            "trainable_params": n_trainable,
            "total_instances": total_instances,
        })
        try:
            result = self._run_epochs(
                torch, ops, pred, net, optimizer, scheduler, train_groups,
                val_groups, epochs, batch_size, prompt_type, freeze_image_encoder,
                device, total_instances, patience, seed, base_file, out_path, tracker,
            )
        finally:
            tracker.end()
        return result

    def _run_epochs(
        self, torch, ops, pred, net, optimizer, scheduler, train_groups,
        val_groups, epochs, batch_size, prompt_type, freeze_image_encoder,
        device, total_instances, patience, seed, base_file, out_path, tracker,
    ) -> dict:
        from .early_stop import EarlyStopper

        rng = random.Random(seed)  # reproducible per-epoch shuffle
        stopper = EarlyStopper(patience)
        best_state = None  # CPU snapshot of the best-val epoch; None ⇒ save last
        has_val = bool(val_groups)

        for epoch in range(1, epochs + 1):
            if self.stop_training:
                break
            rng.shuffle(train_groups)
            epoch_loss, seen, accum = 0.0, 0, 0
            optimizer.zero_grad()

            for group in train_groups:
                if self.stop_training:
                    break
                inst_losses = self._image_instance_losses(
                    torch, ops, pred, group, prompt_type,
                    freeze_image_encoder, device, train=True,
                )
                if not inst_losses:
                    continue

                # batch_size = number of IMAGES to accumulate before an
                # optimizer step (gradient accumulation over images).
                image_loss = torch.stack(inst_losses).mean() / max(1, batch_size)
                image_loss.backward()
                epoch_loss += image_loss.detach().item() * max(1, batch_size)
                seen += 1
                accum += 1
                if accum >= batch_size:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    accum = 0

            if accum > 0:  # flush partial accumulation
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            avg = epoch_loss / max(1, seen)
            cur_lr = optimizer.param_groups[0]["lr"]
            metrics = {"train_loss": avg, "lr": cur_lr}

            val_loss = None
            if has_val:
                val_loss = self._validation_loss(
                    torch, ops, pred, net, val_groups, prompt_type,
                    freeze_image_encoder, device,
                )
                metrics["val_loss"] = val_loss

            tracker.log_metrics(metrics, step=epoch)
            if val_loss is None:
                self.progress_signal.emit(
                    f"Epoch {epoch}/{epochs}  train_loss={avg:.4f}  lr={cur_lr:.2e}  (no val set)"
                )
            else:
                self.progress_signal.emit(
                    f"Epoch {epoch}/{epochs}  train_loss={avg:.4f}  "
                    f"val_loss={val_loss:.4f}  lr={cur_lr:.2e}"
                )

            if has_val:
                if stopper.update(val_loss, epoch):
                    # Best epoch so far — snapshot its weights (cloned to CPU) so
                    # we save the best generalizer, not whatever the last epoch
                    # happened to be.
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if stopper.should_stop:
                    self.progress_signal.emit(
                        f"Early stopping: no val_loss improvement for {patience} epochs "
                        f"(best {stopper.best:.4f} @ epoch {stopper.best_epoch})."
                    )
                    break

        result = self._save_and_verify(net, base_file, out_path, state=best_state)
        result.update(stopped=self.stop_training, instances=total_instances)
        tracker.log_metrics({"stopped_early": int(self.stop_training)})
        if has_val and stopper.best_epoch:
            tracker.log_metrics({"best_val_loss": stopper.best, "best_epoch": stopper.best_epoch})
        tracker.log_artifact(out_path)
        if self.stop_training:
            msg = f"Stopped early — saved {'best' if best_state is not None else 'current'} state to {out_path}"
        elif best_state is not None:
            msg = (f"Saved best model (val_loss {stopper.best:.4f} @ epoch "
                   f"{stopper.best_epoch}): {out_path}")
        else:
            msg = f"Saved fine-tuned model: {out_path}"
        self.progress_signal.emit(msg)
        return result

    def _image_instance_losses(
        self, torch, ops, pred, group, prompt_type, freeze_image_encoder,
        device, *, train: bool,
    ):
        """Per-instance focal+dice losses for one image's annotations.

        ``train`` selects the autograd context: the training pass builds the
        graph (encoder features carry grad only when the encoder is unfrozen) and
        backprops once per image; the validation pass runs entirely under
        ``no_grad`` so it never touches the optimizer and costs no extra memory.
        Returns a list of scalar loss tensors — empty when the image has no
        usable instance.

        One forward + one backward **per image** (the caller stacks these and
        backprops once): when the image encoder is trainable every instance's
        decoder graph hangs off the one shared encoder feature graph, so a
        per-instance backward() would free that shared graph and make the next
        instance raise "backward through the graph a second time".
        """
        image, instances = group.load()
        h, w = image.shape[:2]
        # Compute encoder features with grad only when we're training AND the
        # encoder is unfrozen; validation and frozen-encoder training don't.
        im_t = self._set_image(pred, image, with_feature_grad=train and not freeze_image_encoder)

        losses = []
        grad_ctx = torch.enable_grad() if train else torch.no_grad()
        with grad_ctx:
            for inst in instances:
                bbox = mask_to_xyxy(inst["mask"])
                if bbox is None:
                    continue
                prompt = self._prompt_kwargs(prompt_type, inst["mask"], bbox)
                pm, _ = pred.prompt_inference(im_t, multimask_output=False, **prompt)
                # Map the low-res logits back to the original image with the SAME
                # transform inference uses (SAM2Predictor.postprocess →
                # ops.scale_masks(padding=False)). SAM2 letterboxes the image
                # (resize-min-ratio + pad bottom/right) and scale_masks crops that
                # padding before upsampling. A naive interpolate over the full
                # low-res mask instead bakes the padding region into the target,
                # so the decoder learns masks shifted by the pad — the downward
                # shift seen on non-square images (issue #73 testing).
                logits = ops.scale_masks(pm[:1].unsqueeze(0).float(), (h, w), padding=False)
                target = torch.from_numpy(inst["mask"].astype(np.float32))[None, None].to(device)
                losses.append(_focal_dice_loss(logits, target))
        del image  # bound memory: encoder features recomputed per epoch
        return losses

    def _validation_loss(
        self, torch, ops, pred, net, val_groups, prompt_type,
        freeze_image_encoder, device,
    ) -> float:
        """Mean per-image instance loss over the held-out groups, no grad.

        Runs with ``net.eval()`` so a train-mode image encoder (``_apply_freeze``
        calls ``image_encoder.train()`` when it's unfrozen) doesn't bias the
        measurement; the encoder's train mode is restored afterwards so the next
        epoch trains exactly as before.
        """
        net.eval()
        total, seen = 0.0, 0
        for group in val_groups:
            if self.stop_training:
                break
            losses = self._image_instance_losses(
                torch, ops, pred, group, prompt_type,
                freeze_image_encoder, device, train=False,
            )
            if not losses:
                continue
            total += torch.stack(losses).mean().item()
            seen += 1
        if not freeze_image_encoder:
            net.image_encoder.train()
        return total / max(1, seen)

    def _set_image(self, pred, image: np.ndarray, with_feature_grad: bool):
        """Preprocess ``image`` and compute encoder features into the predictor.

        Sets ``pred.batch`` explicitly so ``_prepare_prompts`` maps bbox/point
        prompts from this image's original pixel size into model coordinates —
        a stale ``batch`` from a different-sized image would mis-scale prompts.

        ``with_feature_grad`` keeps the encoder features in the autograd graph
        (only when training an unfrozen encoder); frozen-encoder training and the
        validation pass compute them under ``no_grad`` to save memory.
        """
        import torch

        pred.setup_source(image)
        im_t = None
        for batch in pred.dataset:
            im_t = pred.preprocess(batch[1])
            break
        # (paths, im0s, ...) — prompt_inference reads self.batch[1][0].shape for orig H,W.
        pred.batch = (None, [image], None)
        # Explicit grad context (not the ambient mode): the unfrozen-encoder
        # training path must build the feature graph even if a prior no_grad
        # val pass left grad disabled, and the val pass must stay grad-free.
        with (torch.enable_grad() if with_feature_grad else torch.no_grad()):
            pred.features = pred.get_im_features(im_t)
        return im_t

    @staticmethod
    def _prompt_kwargs(prompt_type: str, mask: np.ndarray, bbox):
        if prompt_type == "bbox":
            return {"bboxes": [bbox]}
        pt = mask_to_point(mask)
        return {"points": [pt], "labels": [1]}

    # -- checkpoint ----------------------------------------------------------

    def _save_and_verify(self, net, base_file: str, out_path: str, state=None) -> dict:
        """Save fine-tuned weights as ``{"model": state_dict}`` and prove they
        reload through ``SAM(out_path)``.

        ``state`` is the best-val CPU snapshot when early stopping / best-epoch
        selection picked one; when ``None`` (no val set, or no improvement) the
        live ``net`` state is saved, preserving the original last-epoch behaviour.

        Ultralytics' ``_load_checkpoint`` reads only the nested ``"model"`` key
        (a tensor dict) and rebuilds the architecture from the filename suffix,
        so a pure state_dict is all we need — no need to ``torch.load`` (and
        unpickle) the base file. Because ``net`` is the same ``SAM2Model`` class
        Ultralytics instantiates, the keys match exactly, sidestepping the
        "Unexpected key(s)" reload failures (facebookresearch/sam2#337).
        """
        import torch
        from ultralytics import SAM

        token = os.path.basename(base_file)
        if not os.path.basename(out_path).endswith(token):
            raise ValueError(
                f"Fine-tuned checkpoint name must end with '{token}' so "
                f"Ultralytics can pick the right architecture; got "
                f"'{os.path.basename(out_path)}'. Use make_custom_filename()."
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        net_cpu_state = state if state is not None else {
            k: v.detach().cpu() for k, v in net.state_dict().items()
        }
        torch.save({"model": net_cpu_state}, out_path)

        # Round-trip verification: load + one forward. Failing here is loud by design.
        verify = SAM(out_path)
        verify(
            (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
            bboxes=[[8, 8, 56, 56]], verbose=False,
        )
        return {"out_path": out_path, "verified": True}
