"""Tests for SAM 2 fine-tuning (issue bnsreenu#73).

Most tests are CI-friendly: pure geometry/loss/dataset helpers that need no
model weights or GPU. The full train→save→reload round trip is gated behind
``SAM_TRAIN_E2E=1`` and the presence of cached weights, since it needs the
~75 MB ``sam2_t.pt`` and is realistically GPU-only.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from src.digitalsreeni_image_annotator.training.sam_trainer import (
    SAMFineTuner,
    SampleGroup,
    bbox_to_mask,
    list_custom_models,
    make_custom_filename,
    mask_to_point,
    mask_to_xyxy,
    polygon_to_mask,
)


class _SpyTracker:
    """Records the lifecycle calls a trainer makes, in order."""

    active = True

    def __init__(self):
        self.calls = []

    def set_log(self, log):
        self.calls.append("set_log")

    def set_run_url_callback(self, callback):
        pass  # not part of the lifecycle assertion

    def start(self, params=None):
        self.calls.append("start")
        return True

    def log_metrics(self, metrics, step=None):
        self.calls.append("log_metrics")

    def log_artifact(self, path):
        self.calls.append("log_artifact")

    def end(self):
        self.calls.append("end")


def _stub_trainer_internals(monkeypatch, finetuner, run_epochs):
    """Replace the heavy model-building steps of ``train()`` so its tracker
    orchestration can be exercised without weights/GPU. ``run_epochs`` is the
    body to run between ``tracker.start()`` and ``tracker.end()``."""
    import torch

    net = torch.nn.Linear(1, 1)  # a real param so AdamW + device probing work

    class _FakePred:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(
        finetuner, "_build_predictor",
        lambda base: (object(), _FakePred(net), "base.pt"),
    )
    monkeypatch.setattr(
        finetuner, "_apply_freeze",
        lambda net, freeze: (list(net.parameters()), 1),
    )
    monkeypatch.setattr(finetuner, "_run_epochs", run_epochs)


class TestMlflowOrchestration:
    """The trainer must drive the tracker lifecycle (ADR-027, always-on)."""

    def _train(self, finetuner, tracker, out_path):
        finetuner.train(
            "SAM 2 tiny", [SimpleNamespace(n_instances=1)],
            epochs=1, lr=1e-4, batch_size=1, freeze_image_encoder=True,
            prompt_type="bbox", out_path=out_path, tracker=tracker,
        )

    def test_train_drives_tracker_lifecycle(self, monkeypatch, temp_dir):
        ft = SAMFineTuner()
        spy = _SpyTracker()

        def run_epochs(*a, **k):
            tracker = a[-1]  # last positional arg is the tracker
            tracker.log_metrics({"loss": 0.1}, step=1)
            return {"checkpoint": a[-2]}

        _stub_trainer_internals(monkeypatch, ft, run_epochs)
        self._train(ft, spy, os.path.join(temp_dir, "out.pt"))

        # set_log wires the progress sink, then the run opens, logs, and closes.
        assert spy.calls == ["set_log", "start", "log_metrics", "end"]

    def test_train_ends_tracker_even_when_epochs_raise(self, monkeypatch, temp_dir):
        ft = SAMFineTuner()
        spy = _SpyTracker()

        def boom(*a, **k):
            raise RuntimeError("epoch blew up")

        _stub_trainer_internals(monkeypatch, ft, boom)
        with pytest.raises(RuntimeError):
            self._train(ft, spy, os.path.join(temp_dir, "out.pt"))

        # The run must still be closed (the finally: tracker.end() contract).
        assert spy.calls[-1] == "end"
        assert spy.calls == ["set_log", "start", "end"]


class TestValPassAndBestCheckpoint:
    """The split path (issue bnsreenu#85): each epoch logs val_loss + lr, the
    best-val checkpoint is saved (not the last epoch), and patience stops early.
    Drives ``_run_epochs`` directly with light stubs — no weights/GPU."""

    class _RecordingTracker:
        active = True

        def __init__(self):
            self.metrics = []  # (metrics_dict, step)

        def set_log(self, log):
            pass

        def set_run_url_callback(self, cb):
            pass

        def start(self, params=None):
            return True

        def log_metrics(self, metrics, step=None):
            self.metrics.append((metrics, step))

        def log_artifact(self, path):
            pass

        def end(self):
            pass

    def test_logs_val_saves_best_and_early_stops(self, monkeypatch):
        import torch

        ft = SAMFineTuner()
        net = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        # One param-dependent loss per image so backward()/step() actually move
        # the weights — lets us prove the saved checkpoint is best, not last.
        monkeypatch.setattr(ft, "_image_instance_losses", lambda *a, **k: [net.weight.sum()])

        # Val improves through epoch 2 then worsens; patience=2 stops at epoch 4.
        val_seq = iter([0.5, 0.3, 0.4, 0.45, 0.99, 0.99])
        best_weight = {}

        def fake_val(*a, **k):
            v = next(val_seq)
            if abs(v - 0.3) < 1e-9:  # snapshot the weights at the eventual best epoch
                best_weight["w"] = net.weight.detach().clone()
            return v

        monkeypatch.setattr(ft, "_validation_loss", fake_val)

        saved = {}

        def fake_save(net_, base_file, out_path, state=None):
            saved["state"] = state
            return {"out_path": out_path}

        monkeypatch.setattr(ft, "_save_and_verify", fake_save)

        tracker = self._RecordingTracker()
        ft._run_epochs(
            torch, None, None, net, optimizer, None,
            [object(), object()], [object()],            # train_groups, val_groups
            10, 1, "bbox", True, torch.device("cpu"),    # epochs, bs, prompt, freeze, device
            2, 2, 0,                                     # total_instances, patience, seed
            "base_sam2_t.pt", "out_sam2_t.pt", tracker,
        )

        # val_loss + lr logged every epoch; the run early-stopped after epoch 4.
        val_steps = [step for (m, step) in tracker.metrics if "val_loss" in m]
        assert val_steps == [1, 2, 3, 4]
        assert all("lr" in m for (m, step) in tracker.metrics if step in (1, 2, 3, 4))
        # Best (epoch-2) weights saved, NOT the last epoch's.
        assert saved["state"] is not None
        assert torch.allclose(saved["state"]["weight"], best_weight["w"])
        assert not torch.allclose(saved["state"]["weight"], net.weight.detach())

    def test_no_val_set_saves_final_state(self, monkeypatch):
        import torch

        ft = SAMFineTuner()
        net = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        monkeypatch.setattr(ft, "_image_instance_losses", lambda *a, **k: [net.weight.sum()])
        monkeypatch.setattr(
            ft, "_validation_loss",
            lambda *a, **k: pytest.fail("validation pass ran with no val set"),
        )

        saved = {}

        def fake_save(net_, base_file, out_path, state=None):
            saved["state"] = state
            return {"out_path": out_path}

        monkeypatch.setattr(ft, "_save_and_verify", fake_save)

        tracker = self._RecordingTracker()
        ft._run_epochs(
            torch, None, None, net, optimizer, None,
            [object()], [],                              # train_groups, NO val_groups
            3, 1, "bbox", True, torch.device("cpu"),     # epochs, bs, prompt, freeze, device
            5, 20, 0,                                    # total_instances, patience, seed
            "base_sam2_t.pt", "out_sam2_t.pt", tracker,
        )

        # No val ⇒ state=None ⇒ _save_and_verify saves the live (final) net state.
        assert saved["state"] is None
        assert not any("val_loss" in m for (m, step) in tracker.metrics)
        assert any("train_loss" in m for (m, step) in tracker.metrics)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── geometry helpers ─────────────────────────────────────────────────────────

class TestGeometry:
    def test_polygon_to_mask_fills_interior(self):
        mask = polygon_to_mask([10, 10, 40, 10, 40, 40, 10, 40], 50, 50)
        assert mask.dtype == bool and mask.shape == (50, 50)
        assert mask[25, 25] and not mask[5, 5]

    def test_bbox_to_mask(self):
        mask = bbox_to_mask([10, 10, 20, 20], 50, 50)  # x,y,w,h
        assert mask[15, 15] and not mask[45, 45]

    def test_mask_to_xyxy_tight(self):
        mask = np.zeros((50, 50), bool)
        mask[10:31, 5:26] = True
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        assert (x1, y1, x2, y2) == (5, 10, 25, 30)

    def test_mask_to_xyxy_empty_is_none(self):
        assert mask_to_xyxy(np.zeros((10, 10), bool)) is None

    def test_mask_to_point_inside(self):
        mask = np.zeros((60, 60), bool)
        mask[20:41, 20:41] = True
        x, y = mask_to_point(mask)
        assert mask[int(y), int(x)]  # point lands inside the object

    def test_polygon_roundtrips_through_xyxy(self):
        mask = polygon_to_mask([10, 10, 40, 10, 40, 40, 10, 40], 50, 50)
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        assert x1 >= 9 and y1 >= 9 and x2 <= 41 and y2 <= 41


# ── checkpoint naming (architecture-selection invariant) ─────────────────────

class TestCustomNaming:
    @pytest.mark.parametrize("base,token", [
        ("SAM 2 tiny", "sam2_t.pt"),
        ("SAM 2.1 base", "sam2.1_b.pt"),
        ("SAM 2.1 large", "sam2.1_l.pt"),
    ])
    def test_filename_keeps_base_token(self, base, token):
        # Ultralytics build_sam selects the architecture by ckpt.endswith(token);
        # a fine-tuned name MUST keep that suffix or SAM(path) can't load it.
        path = make_custom_filename(base, "my cool run!!")
        assert os.path.basename(path).endswith(token)

    def test_filename_sanitises_label(self):
        path = make_custom_filename("SAM 2 tiny", "a/b c:*?")
        name = os.path.basename(path)
        assert "/" not in name and ":" not in name and "*" not in name

    def test_filename_handles_empty_label(self):
        path = make_custom_filename("SAM 2 tiny", "")
        assert os.path.basename(path) == "finetuned_sam2_t.pt"

    def test_list_custom_models_returns_dict(self):
        assert isinstance(list_custom_models(), dict)


# ── SampleGroup lazy rasterisation ───────────────────────────────────────────

class TestSampleGroup:
    def test_load_rasterises_specs(self):
        img = np.zeros((50, 50, 3), np.uint8)
        specs = [
            {"segmentation": [10, 10, 40, 10, 40, 40, 10, 40]},
            {"bbox": [5, 5, 10, 10]},
        ]
        group = SampleGroup(lambda: img.copy(), specs)
        assert group.n_instances == 2
        out_img, instances = group.load()
        assert out_img.shape == (50, 50, 3)
        assert len(instances) == 2
        assert all(inst["mask"].dtype == bool for inst in instances)

    def test_load_drops_empty_masks(self):
        img = np.zeros((50, 50, 3), np.uint8)
        # Box entirely outside the image → no in-bounds pixels → dropped.
        group = SampleGroup(lambda: img.copy(), [{"bbox": [200, 200, 10, 10]}])
        _, instances = group.load()
        assert instances == []


# ── loss ─────────────────────────────────────────────────────────────────────

class TestLoss:
    def test_focal_dice_lower_when_correct(self):
        torch = pytest.importorskip("torch")
        from src.digitalsreeni_image_annotator.training.sam_trainer import _focal_dice_loss

        target = torch.zeros(1, 1, 16, 16)
        target[..., 4:12, 4:12] = 1.0
        good = (target * 12) - 6  # confident-correct logits
        bad = (1 - target) * 12 - 6  # confident-wrong logits
        l_good = _focal_dice_loss(good, target)
        l_bad = _focal_dice_loss(bad, target)
        assert torch.isfinite(l_good) and torch.isfinite(l_bad)
        assert float(l_good) < float(l_bad)


# ── dataset producers ────────────────────────────────────────────────────────

class TestDatasetProducers:
    def test_export_and_reload_folder_roundtrip(self, temp_dir):
        from src.digitalsreeni_image_annotator.io.export_formats import export_sam_dataset
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_folder,
        )

        img = QImage(60, 60, QImage.Format.Format_RGB32)
        img.fill(0xFF202020)
        img_path = os.path.join(temp_dir, "img1.png")
        img.save(img_path)

        all_annotations = {
            "img1.png": {"cell": [{"segmentation": [10, 10, 40, 10, 40, 40, 10, 40]}]}
        }
        out_dir = os.path.join(temp_dir, "dataset")
        _, manifest_path = export_sam_dataset(
            all_annotations, {"cell": 1}, {"img1.png": img_path},
            slices=[], image_slices={}, output_dir=out_dir,
        )
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest["images"]) == 1

        groups = build_groups_from_folder(out_dir)
        assert len(groups) == 1
        _, instances = groups[0].load()
        assert len(instances) == 1

    def test_build_from_project_regular_image(self, temp_dir):
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_project,
        )

        img = QImage(60, 60, QImage.Format.Format_RGB32)
        img.fill(0xFF808080)
        img_path = os.path.join(temp_dir, "img1.png")
        img.save(img_path)

        groups = build_groups_from_project(
            {"img1.png": {"cell": [{"bbox": [10, 10, 20, 20]}]}},
            {"img1.png": img_path}, slices=[], image_slices={},
        )
        assert len(groups) == 1
        image, instances = groups[0].load()
        assert image.shape[:2] == (60, 60) and len(instances) == 1

    def test_build_from_project_skips_unresolvable(self):
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_project,
        )

        groups = build_groups_from_project(
            {"ghost.png": {"cell": [{"bbox": [1, 1, 5, 5]}]}},
            {}, slices=[], image_slices={},
        )
        assert groups == []


# ── ultralytics API-drift guard ──────────────────────────────────────────────

class TestUltralyticsAPI:
    def test_sam2predictor_exposes_forward_methods(self):
        """The native training loop reuses these SAM2Predictor methods; if an
        Ultralytics upgrade removes/renames them, fail loudly here rather than
        mid-training."""
        pytest.importorskip("ultralytics")
        from ultralytics.models.sam.predict import SAM2Predictor

        for name in ("get_im_features", "prompt_inference", "_inference_features",
                     "_prepare_prompts", "set_image"):
            assert hasattr(SAM2Predictor, name), f"SAM2Predictor.{name} missing"

    def test_ops_scale_masks_padding_false_geometry(self):
        """The training loss maps logits back with the same letterbox-aware
        transform inference uses (ops.scale_masks, padding=False). Guard the
        *behavior*, not just the name: with padding=False the crop is
        top-left-anchored, so foreground in the bottom padding band of a
        non-square target is dropped while top content survives. A semantic
        change in Ultralytics (not just a rename) trips this fast test rather
        than only the GPU-gated e2e."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("ultralytics")
        from ultralytics.utils import ops

        assert hasattr(ops, "scale_masks")
        # Target 600x1000 (landscape) -> ~bottom 40% of a 256-tall mask is padding.
        top = torch.zeros(1, 1, 256, 256)
        top[..., 0:40, :] = 1.0
        bottom = torch.zeros(1, 1, 256, 256)
        bottom[..., 215:256, :] = 1.0
        out_top = ops.scale_masks(top, (600, 1000), padding=False)
        out_bottom = ops.scale_masks(bottom, (600, 1000), padding=False)
        assert out_top.sum() > 0, "top content must survive the crop"
        assert out_bottom.sum() == 0, "bottom padding band must be cropped (padding=False)"


# ── opt-in end-to-end (needs weights; realistically GPU) ─────────────────────

@pytest.mark.skipif(
    os.environ.get("SAM_TRAIN_E2E") != "1",
    reason="set SAM_TRAIN_E2E=1 to run the full train→save→reload test (needs weights)",
)
def test_end_to_end_train_save_reload(temp_dir):
    pytest.importorskip("ultralytics")
    from src.digitalsreeni_image_annotator.training.sam_trainer import (
        SAM_MODELS_DIR,
        SAMFineTuner,
    )

    if not os.path.exists(os.path.join(SAM_MODELS_DIR, "sam2_t.pt")):
        pytest.skip("sam2_t.pt not cached")

    def make(seed):
        rng = np.random.RandomState(seed)
        img = (rng.rand(120, 120, 3) * 255).astype(np.uint8)
        cy, cx = rng.randint(40, 80, 2)
        yy, xx = np.ogrid[:120, :120]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < 25 ** 2
        img[mask] = [220, 30, 30]
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        return SampleGroup(
            lambda im=img: im.copy(),
            [{"bbox": [x1, y1, x2 - x1, y2 - y1]}],
        )

    out = os.path.join(temp_dir, "e2e_sam2_t.pt")
    res = SAMFineTuner().train(
        "SAM 2 tiny", [make(i) for i in range(2)],
        epochs=1, lr=1e-4, batch_size=1, freeze_image_encoder=True,
        prompt_type="bbox", out_path=out,
    )
    assert res["verified"] and os.path.exists(out)

    from ultralytics import SAM
    SAM(out)((np.random.rand(120, 120, 3) * 255).astype(np.uint8),
             bboxes=[[40, 40, 80, 80]], verbose=False)


@pytest.mark.skipif(
    os.environ.get("SAM_TRAIN_E2E") != "1",
    reason="set SAM_TRAIN_E2E=1 to run the encoder-path multi-instance test",
)
def test_encoder_path_multi_instance_per_image(temp_dir):
    """Regression: encoder fine-tuning (freeze_image_encoder=False) on images
    with >1 instance used to crash with 'backward through the graph a second
    time' because all instances shared the one encoder feature graph. The
    engine now backprops once per image."""
    pytest.importorskip("ultralytics")
    from src.digitalsreeni_image_annotator.training.sam_trainer import (
        SAM_MODELS_DIR,
        SAMFineTuner,
    )

    if not os.path.exists(os.path.join(SAM_MODELS_DIR, "sam2_t.pt")):
        pytest.skip("sam2_t.pt not cached")

    def two_instance_image(seed):
        rng = np.random.RandomState(seed)
        img = (rng.rand(140, 140, 3) * 255).astype(np.uint8)
        specs = []
        for cy, cx in [(40, 40), (95, 95)]:
            yy, xx = np.ogrid[:140, :140]
            m = (yy - cy) ** 2 + (xx - cx) ** 2 < 18 ** 2
            img[m] = [30, 220, 30]
            x1, y1, x2, y2 = mask_to_xyxy(m)
            specs.append({"bbox": [x1, y1, x2 - x1, y2 - y1]})
        return SampleGroup(lambda im=img: im.copy(), specs)

    out = os.path.join(temp_dir, "enc_sam2_t.pt")
    res = SAMFineTuner().train(
        "SAM 2 tiny", [two_instance_image(i) for i in range(2)],
        epochs=1, lr=1e-5, batch_size=1, freeze_image_encoder=False,
        prompt_type="bbox", out_path=out,
    )
    assert res["verified"] and res["instances"] == 4


@pytest.mark.skipif(
    os.environ.get("SAM_TRAIN_E2E") != "1",
    reason="set SAM_TRAIN_E2E=1 to run the landscape mask-shift regression",
)
def test_landscape_no_mask_shift(temp_dir):
    """Regression for the downward mask shift (issue #73 GUI testing).

    SAM2 letterboxes (pad bottom/right) and inference crops the padding via
    ops.scale_masks(padding=False). The training loss must use the SAME
    transform; a naive interpolate baked the padding into the target and the
    decoder learned masks shifted down. This must use a NON-square image (square
    images have zero padding and so never exposed the bug).
    """
    pytest.importorskip("ultralytics")
    import cv2
    from ultralytics import SAM

    from src.digitalsreeni_image_annotator.training.sam_trainer import (
        SAM_MODELS_DIR,
        SAMFineTuner,
    )

    if not os.path.exists(os.path.join(SAM_MODELS_DIR, "sam2_t.pt")):
        pytest.skip("sam2_t.pt not cached")

    H, W = 600, 1000  # landscape -> SAM2 pads the bottom

    def disk(seed, cy, cx):
        rng = np.random.RandomState(seed)
        img = (rng.rand(H, W, 3) * 50).astype(np.uint8)
        yy, xx = np.ogrid[:H, :W]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < 60 ** 2
        img[m] = [230, 40, 40]
        return img, m

    groups = []
    for i in range(4):
        img, m = disk(i, 460 + (i % 2) * 40, 300 + i * 120)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        groups.append(SampleGroup(lambda im=img: im.copy(), [{"segmentation": cnts[0].flatten().tolist()}]))

    out = os.path.join(temp_dir, "landscape_sam2_t.pt")
    SAMFineTuner().train(
        "SAM 2 tiny", groups, epochs=8, lr=1e-4, batch_size=2,
        freeze_image_encoder=True, prompt_type="bbox", out_path=out,
    )

    # Evaluate on a fresh image, object near the bottom (worst case for the shift).
    img, m = disk(99, 480, 520)
    x1, y1, x2, y2 = mask_to_xyxy(m)
    gt_cy = float(np.where(m)[0].mean())

    r = SAM(out)(img, bboxes=[[x1, y1, x2, y2]], verbose=False)
    pm = r[0].masks.data.cpu().numpy()[0].astype(bool)
    ys, xs = np.where(pm)
    assert xs.size > 0, "fine-tuned model produced an empty mask"
    pred_cy = ys.mean()
    iou = (pm & m).sum() / (pm | m).sum()

    assert abs(pred_cy - gt_cy) < 25, f"vertical shift {pred_cy - gt_cy:.1f}px (regression)"
    assert iou > 0.7, f"IoU {iou:.3f} too low"
