"""Unit tests for ``SAM3Utils`` (issue #50, ADR-038).

The real ``sam3.pt`` checkpoint is ~3.45 GB and gated on Hugging Face, so it
is NOT present here or in CI. Every test therefore stubs the predictor — the
real ``SAM3SemanticPredictor`` is never instantiated. We only exercise our
own marshalling / filtering / polygon-conversion glue around it.
"""

import numpy as np
import pytest

from digitalsreeni_image_annotator.inference.sam3_utils import SAM3Utils


# --- fakes mimicking the Ultralytics Results / tensor surface ---------------

class _FakeTensor:
    """Mimics a torch tensor's ``.cpu().numpy()`` chain over a numpy array."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeMasks:
    def __init__(self, masks_np):
        self.data = _FakeTensor(masks_np)


class _FakeBoxes:
    def __init__(self, conf_np, xyxy_np):
        self.conf = _FakeTensor(conf_np)
        self.xyxy = _FakeTensor(xyxy_np)


class _FakeResults:
    def __init__(self, masks_np, conf_np, xyxy_np):
        self.masks = _FakeMasks(masks_np) if masks_np is not None else None
        self.boxes = (
            _FakeBoxes(conf_np, xyxy_np) if masks_np is not None else None
        )


class _FakePredictor:
    """Records calls; returns a preset Results (object or list)."""

    def __init__(self, results):
        self._results = results
        self.set_image_calls = []
        self.text_calls = []

    def set_image(self, image_np):
        self.set_image_calls.append(image_np)

    def __call__(self, text=None):
        self.text_calls.append(text)
        return self._results


def _square_mask(h=64, w=64, x0=5, y0=5, size=20):
    """A filled square well above the area-10 contour floor."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y0 + size, x0:x0 + size] = 1
    return m


def _tiny_mask(h=64, w=64):
    """A 2x2 blob — area 4, below the area-10 floor -> no polygon."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[0:2, 0:2] = 1
    return m


def _cfg(box_thr=0.3, phrases=("cell",)):
    return {"name": "cell", "phrases": list(phrases),
            "box_thr": box_thr, "txt_thr": 0.25, "nms_thr": 0.5}


def _blank_qimage():
    from PyQt6.QtGui import QImage
    img = QImage(64, 64, QImage.Format.Format_RGB888)
    img.fill(0)
    return img


@pytest.fixture
def sam3(qt_application):
    """A fresh (unloaded) SAM3Utils. ``qt_application`` so ``_run_sync`` has
    a GUI thread to run against."""
    return SAM3Utils()


def _load_fake(s, results):
    s._predictor = _FakePredictor(results)
    s.loaded = True
    return s._predictor


# --- detect_text: output shape ---------------------------------------------

def test_detect_text_returns_instance_shape(sam3):
    masks = np.stack([_square_mask(x0=5), _square_mask(x0=35)])
    conf = np.array([0.9, 0.8], dtype=np.float32)
    xyxy = np.array([[5, 5, 25, 25], [35, 5, 55, 25]], dtype=np.float32)
    _load_fake(sam3, _FakeResults(masks, conf, xyxy))

    out = sam3.detect_text(_blank_qimage(), [_cfg()])

    assert len(out) == 2
    for inst in out:
        assert set(inst) == {"class_name", "score", "segmentation", "bbox"}
        assert inst["class_name"] == "cell"
        assert isinstance(inst["segmentation"], list)
        assert len(inst["segmentation"]) >= 6
        assert len(inst["bbox"]) == 4
    assert out[0]["score"] == pytest.approx(0.9)
    assert out[1]["score"] == pytest.approx(0.8)


def test_detect_text_accepts_results_as_list(sam3):
    # The spike describes a Results object; be defensive about a 1-element
    # list too (the usual Ultralytics predictor return).
    masks = np.stack([_square_mask()])
    conf = np.array([0.9], dtype=np.float32)
    xyxy = np.array([[5, 5, 25, 25]], dtype=np.float32)
    _load_fake(sam3, [_FakeResults(masks, conf, xyxy)])

    out = sam3.detect_text(_blank_qimage(), [_cfg()])

    assert len(out) == 1


# --- detect_text: confidence (box_thr) filtering ----------------------------

def test_detect_text_drops_instance_below_box_thr(sam3):
    masks = np.stack([_square_mask(x0=5), _square_mask(x0=35)])
    conf = np.array([0.9, 0.1], dtype=np.float32)   # 0.1 < box_thr 0.3
    xyxy = np.array([[5, 5, 25, 25], [35, 5, 55, 25]], dtype=np.float32)
    _load_fake(sam3, _FakeResults(masks, conf, xyxy))

    out = sam3.detect_text(_blank_qimage(), [_cfg(box_thr=0.3)])

    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.9)


# --- detect_text: area floor delegated to _mask_to_polygon ------------------

def test_detect_text_small_mask_yields_no_instance(sam3):
    masks = np.stack([_tiny_mask()])
    conf = np.array([0.99], dtype=np.float32)
    xyxy = np.array([[0, 0, 2, 2]], dtype=np.float32)
    _load_fake(sam3, _FakeResults(masks, conf, xyxy))

    out = sam3.detect_text(_blank_qimage(), [_cfg()])

    assert out == []


def test_detect_text_delegates_polygon_to_mask_to_polygon(sam3, monkeypatch):
    import digitalsreeni_image_annotator.inference.sam3_utils as mod

    calls = []
    real = mod._mask_to_polygon

    def spy(mask):
        calls.append(getattr(mask, "shape", None))
        return real(mask)

    monkeypatch.setattr(mod, "_mask_to_polygon", spy)

    masks = np.stack([_square_mask()])
    conf = np.array([0.9], dtype=np.float32)
    xyxy = np.array([[5, 5, 25, 25]], dtype=np.float32)
    _load_fake(sam3, _FakeResults(masks, conf, xyxy))

    out = sam3.detect_text(_blank_qimage(), [_cfg()])

    assert calls  # conversion went through the shared helper
    assert len(out) == 1


# --- detect_text: unloaded model never half-works ---------------------------

def test_detect_text_unloaded_returns_none(sam3):
    # Fresh instance: loaded is False, _predictor is None.
    assert sam3.loaded is False
    assert sam3.detect_text(_blank_qimage(), [_cfg()]) is None


# --- model load: overrides must omit quantize / mode ------------------------

def test_load_overrides_omit_quantize_and_mode(sam3, monkeypatch):
    captured = {}

    class _Recorder:
        def __init__(self, overrides=None, **kwargs):
            captured["overrides"] = overrides
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "ultralytics.models.sam.SAM3SemanticPredictor", _Recorder
    )

    # Pure Python; safe to call directly (no worker thread needed).
    sam3._load_model_blocking()

    assert sam3.loaded is True
    assert sam3._predictor is not None
    ov = captured["overrides"]
    # device is REQUIRED (CPU-fallback safety, mirrors SAMUtils); quantize/mode
    # must be absent (quantize raises in ultralytics 8.4.51, mode is redundant).
    # save/verbose are forced False to suppress the runs/ dir + per-call console
    # summary Ultralytics writes on every detection (verified in the #50 manual
    # GPU run, 2026-07-22 -- accepted without raising).
    assert set(ov) == {"model", "task", "conf", "device", "save", "verbose"}
    assert "quantize" not in ov
    assert "mode" not in ov
    assert ov["save"] is False
    assert ov["verbose"] is False
    assert ov["task"] == "segment"
    assert isinstance(ov["conf"], float)
    assert ov["device"] == sam3._device
    # Constructor gets ONLY overrides=...; no stray kwargs.
    assert captured["kwargs"] == {}


def test_failed_load_leaves_unloaded_and_reraises(sam3, monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("weights corrupt")

    monkeypatch.setattr(
        "ultralytics.models.sam.SAM3SemanticPredictor", _Boom
    )

    with pytest.raises(RuntimeError):
        sam3._load_model_blocking()

    # Never a half-loaded state.
    assert sam3.loaded is False
    assert sam3._predictor is None


# --- weights resolution (no download) ---------------------------------------

def test_weights_available_reflects_resolvable_path(sam3, monkeypatch, tmp_path):
    absent = tmp_path / "absent.pt"
    monkeypatch.setattr(sam3, "_candidate_weight_paths", lambda: [str(absent)])
    assert sam3._resolve_weights_path() is None
    assert sam3.weights_available() is False

    present = tmp_path / "sam3.pt"
    present.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setattr(sam3, "_candidate_weight_paths", lambda: [str(present)])
    assert sam3._resolve_weights_path() == str(present)
    assert sam3.weights_available() is True


def test_env_override_is_a_candidate_path(sam3, monkeypatch, tmp_path):
    present = tmp_path / "custom-sam3.pt"
    present.write_bytes(b"x")
    monkeypatch.setenv("SAM3_MODEL_PATH", str(present))
    assert str(present) in sam3._candidate_weight_paths()
    assert sam3._resolve_weights_path() == str(present)
