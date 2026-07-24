"""Integration tests for the SAM 3 text-prompt workflow (issue #50, ADR-038).

SAM 3 is a new producer plugged into the exact spot the DINO->SAM pipeline
occupies. These build one real ImageAnnotator (offscreen), replace
``window.sam3_utils`` with a stub whose ``detect_text`` returns deterministic
instances, and drive ``window.dino_controller`` — proving the whole downstream
review machinery (temp overlay, accept/reject, batch keying, per-image temp
re-sync, auto-accept) is reused verbatim.

No model weights, no network, no worker threads on the real model.
"""

import pytest


# Two deterministic instances SAM 3 "produced" — both class "cell".
INSTANCES = [
    {"class_name": "cell", "score": 0.9,
     "segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0], "bbox": [1.0, 1.0, 10.0, 10.0]},
    {"class_name": "cell", "score": 0.8,
     "segmentation": [20.0, 20.0, 30.0, 20.0, 30.0, 30.0], "bbox": [20.0, 20.0, 30.0, 30.0]},
]


class _Sam3Stub:
    """Stands in for SAM3Utils. Records detect_text calls via a spy list."""

    def __init__(self, instances):
        self._instances = instances
        self.loaded = True
        self.detect_calls = []

    def weights_available(self):
        return True

    def ensure_loaded(self):
        self.loaded = True

    def detect_text(self, image, class_configs):
        self.detect_calls.append((image, list(class_configs)))
        # Fresh dicts each call (batch reuses across work items).
        return [dict(i) for i in self._instances]

    def unload(self):
        self.loaded = False


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


@pytest.fixture
def sam3_ready(window, monkeypatch):
    """Window primed for SAM 3 text detection with a stubbed producer and the
    picker set to the SAM 3 entry."""
    from PyQt6.QtGui import QImage, QColor
    from PyQt6.QtWidgets import QMessageBox
    from digitalsreeni_image_annotator.inference.sam3_utils import SAM3_MODEL_LABEL
    import digitalsreeni_image_annotator.core.torch_utils as torch_utils

    c = window.dino_controller
    stub = _Sam3Stub(INSTANCES)
    window.sam3_utils = stub

    monkeypatch.setattr(
        c, "_build_dino_class_configs",
        lambda: [{"name": "cell", "phrases": ["cell"],
                  "box_thr": 0.3, "txt_thr": 0.25, "nms_thr": 0.5}],
    )
    monkeypatch.setattr(torch_utils, "maybe_warn_cpu_fallback", lambda *a, **k: None)
    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    img = QImage(64, 64, QImage.Format.Format_RGB888)
    img.fill(0)
    window.current_image = img
    window.image_file_name = "img1.png"
    window.current_slice = None
    window.class_mapping = {"cell": 1}
    window.image_label.class_colors["cell"] = QColor(255, 0, 0)

    # Select SAM 3 in the DINO picker — fires _on_sam3_selected against the stub.
    window.dino_model_selector.setCurrentText(SAM3_MODEL_LABEL)
    return window


def _seed_batch(window, tmp_path):
    """One regular image (real PNG on disk) + one materialised slice."""
    from PyQt6.QtGui import QImage
    from PIL import Image

    reg = tmp_path / "reg1.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(reg)

    qimg = QImage(8, 8, QImage.Format.Format_RGB888)
    qimg.fill(0)

    window.all_images = [
        {"file_name": "reg1.png"},
        {"file_name": "stack.tif", "is_multi_slice": True},
    ]
    window.image_paths = {"reg1.png": str(reg)}
    window.image_slices = {"stack": [("stack_T0_Z0", qimg)]}
    return ["reg1.png", "stack_T0_Z0"]


# --- model picker wiring ----------------------------------------------------

def test_selecting_sam3_enables_detect(sam3_ready):
    window = sam3_ready
    assert window.sam3_utils.loaded is True
    assert window.btn_detect_single.isEnabled()
    assert window.btn_detect_batch.isEnabled()
    assert window.dino_controller._is_sam3_selected() is True


def test_sam3_gated_download_status_when_weights_absent(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from digitalsreeni_image_annotator.inference.sam3_utils import SAM3_MODEL_LABEL

    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))

    class _NoWeights(_Sam3Stub):
        def weights_available(self):
            return False

        def ensure_loaded(self):
            raise AssertionError("must not load when weights are unavailable")

    window.sam3_utils = _NoWeights(INSTANCES)
    window.dino_model_selector.setCurrentText(SAM3_MODEL_LABEL)

    # Gated: no load attempted, detect buttons stay disabled, status prompts
    # the user to fetch the weights.
    assert not window.btn_detect_single.isEnabled()
    assert not window.btn_detect_batch.isEnabled()
    assert "sam3.pt" in window.lbl_dino_status.text()


# --- single detection -------------------------------------------------------

def test_single_review_attaches_sam3_temp(sam3_ready):
    window = sam3_ready
    c = window.dino_controller
    # dino_batch_mode left at the default "Review before accepting".

    c.run_dino_detection_single()

    temp = window.image_label.temp_annotations
    assert len(temp) == 2
    assert all(t["source"] == "sam3" for t in temp)
    assert all(t["temp"] is True for t in temp)
    assert all(t["category_name"] == "cell" for t in temp)
    # Review mode commits nothing to the canvas.
    assert "cell" not in window.image_label.annotations
    # The stub producer was actually exercised.
    assert len(window.sam3_utils.detect_calls) == 1


def test_single_accept_commits_and_clears(sam3_ready):
    window = sam3_ready
    c = window.dino_controller
    c.run_dino_detection_single()
    assert len(window.image_label.temp_annotations) == 2

    c.accept_dino_results()

    cells = window.image_label.annotations.get("cell", [])
    assert len(cells) == 2
    assert all(a["source"] == "sam3" for a in cells)
    assert window.image_label.temp_annotations == []
    assert window.all_annotations.get("img1.png", {}).get("cell")


def test_single_auto_accept_commits_directly(sam3_ready):
    window = sam3_ready
    c = window.dino_controller
    window.dino_batch_mode.setCurrentText("Auto-accept all detections")

    c.run_dino_detection_single()

    # Single path honors the (batch-labelled) auto-accept dropdown.
    assert window.image_label.temp_annotations == []
    cells = window.image_label.annotations.get("cell", [])
    assert len(cells) == 2
    assert cells[0]["number"] == 1
    assert cells[1]["number"] == 2
    assert cells[0]["source"] == "sam3"


# --- batch detection --------------------------------------------------------

def test_batch_review_keys_per_image_and_slice(sam3_ready, tmp_path, monkeypatch):
    window = sam3_ready
    c = window.dino_controller
    names = _seed_batch(window, tmp_path)
    window.dino_batch_mode.setCurrentText("Review before accepting")
    monkeypatch.setattr(c, "_show_dino_batch_review", lambda: None)

    c.run_dino_detection_batch()

    assert set(window.dino_batch_results) == set(names)
    for res in window.dino_batch_results.values():
        assert len(res) == 2
        assert all(r["source"] == "sam3" for r in res)
        assert all("segmentation" in r for r in res)
    # detect_text ran once per work item.
    assert len(window.sam3_utils.detect_calls) == len(names)


def test_batch_temp_resync_swaps_on_switch(sam3_ready, tmp_path, monkeypatch):
    window = sam3_ready
    c = window.dino_controller
    _seed_batch(window, tmp_path)
    window.dino_batch_mode.setCurrentText("Review before accepting")
    monkeypatch.setattr(c, "_show_dino_batch_review", lambda: None)
    c.run_dino_detection_batch()

    # Switch to the regular image -> its stored set becomes the live temp.
    window.image_file_name = "reg1.png"
    window.current_slice = None
    c._refresh_dino_temp_for_current()
    assert len(window.image_label.temp_annotations) == 2
    assert all(t["source"] == "sam3" for t in window.image_label.temp_annotations)

    # Switch to the slice -> the temp set swaps (no bleed across the switch).
    window.current_slice = "stack_T0_Z0"
    c._refresh_dino_temp_for_current()
    assert len(window.image_label.temp_annotations) == 2

    # Switch to an image with no pending results -> temp cleared.
    window.current_slice = None
    window.image_file_name = "unrelated.png"
    c._refresh_dino_temp_for_current()
    assert window.image_label.temp_annotations == []


# --- DINO fallback path is untouched ---------------------------------------

def test_inflight_guard_absorbs_busy(sam3_ready, tmp_path, monkeypatch):
    """A re-entrant SAM 3 call raises InferenceBusyError; both the single and
    batch paths must absorb it (skip cleanly) rather than crash (ADR-013)."""
    from digitalsreeni_image_annotator.inference.sam3_utils import (
        InferenceBusyError,
    )

    window = sam3_ready
    c = window.dino_controller

    def _busy(image, class_configs):
        raise InferenceBusyError("busy")

    window.sam3_utils.detect_text = _busy

    # Single: no crash, nothing attached.
    c.run_dino_detection_single()
    assert window.image_label.temp_annotations == []

    # Batch: no crash either (each work item's busy is caught + skipped).
    _seed_batch(window, tmp_path)
    window.dino_batch_mode.setCurrentText("Review before accepting")
    monkeypatch.setattr(c, "_show_dino_batch_review", lambda: None)
    c.run_dino_detection_batch()  # must not raise


def test_dino_path_does_not_call_sam3(sam3_ready, monkeypatch):
    window = sam3_ready
    c = window.dino_controller

    # Point the picker at a real DINO entry; wire the legacy two-stage stubs.
    window.dino_model_loaded = True
    window.sam_utils.current_sam_model = "SAM 2 tiny"
    monkeypatch.setattr(c, "_ensure_dino_model_downloaded", lambda name: True)
    window.dino_model_selector.setCurrentText("grounding-dino-base")
    monkeypatch.setattr(
        window.dino_utils, "detect",
        lambda *a, **k: [{"class_name": "cell", "score": 0.9, "bbox": [1, 1, 10, 10]}],
    )
    monkeypatch.setattr(
        window.sam_utils, "apply_sam_predictions_batch",
        lambda img, boxes: [{"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0]}
                            for _ in boxes],
    )

    c.run_dino_detection_single()

    # SAM 3 stub must be untouched by the DINO path.
    assert window.sam3_utils.detect_calls == []
    # And the legacy path produced a DINO-sourced temp annotation.
    temp = window.image_label.temp_annotations
    assert len(temp) == 1
    assert temp[0]["source"] == "dino"
