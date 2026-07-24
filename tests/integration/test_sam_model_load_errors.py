"""SAMController.change_sam_model error-dialog tests (issue #34).

A model load failure must surface a ``QMessageBox.critical`` and reset the model
selector to index 0. An out-of-memory failure gets a tailored "pick a smaller
model" message; any other failure keeps the generic download/torch guidance.
Never calls the real ``SAMUtils.change_sam_model`` (it downloads 40-400 MB of
weights) — the method is monkeypatched to raise.
"""

import pytest
from PyQt6.QtWidgets import QMessageBox


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _raiser(exc):
    def _fn(_name):
        raise exc
    return _fn


def test_oom_load_shows_friendly_dialog(window, monkeypatch):
    captured = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: captured.append(a))
    )
    monkeypatch.setattr(
        window.sam_utils, "change_sam_model",
        _raiser(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
    )

    window.sam_controller.change_sam_model("SAM 2 large")

    assert captured, "expected a QMessageBox.critical to be shown"
    _parent, _title, text = captured[0][:3]
    assert "smaller" in text.lower() and "memory" in text.lower()
    assert window.sam_model_selector.currentIndex() == 0


def test_generic_load_failure_keeps_existing_dialog(window, monkeypatch):
    captured = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: captured.append(a))
    )
    monkeypatch.setattr(
        window.sam_utils, "change_sam_model",
        _raiser(RuntimeError("404 not found")),
    )

    window.sam_controller.change_sam_model("SAM 2 large")

    assert captured, "expected a QMessageBox.critical to be shown"
    _parent, _title, text = captured[0][:3]
    assert "404" in text or "downloadable" in text
    assert "smaller" not in text.lower()  # generic message, not the OOM one
    assert window.sam_model_selector.currentIndex() == 0
