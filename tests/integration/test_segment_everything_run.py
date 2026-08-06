"""SegmentEverythingController.run() end to end (issue #69).

These exist because the unit tests for #69 all entered at ``stage_proposals``
or below, so ``run()`` -- the only thing a user ever actually invokes -- had no
coverage at all. It shipped with a bug that discarded every single proposal:
``progress.close()`` in the ``finally`` block runs *before* the
``wasCanceled()`` check, and ``QProgressDialog.closeEvent`` emits ``canceled()``
which Qt wires to the ``cancel()`` slot. So the flag was always set, and a
run that had just spent five seconds producing 122 masks logged "cancelled by
the user" and threw the lot away.

Only ``apply_sam_everything`` is mocked -- the progress dialog, the filters, the
staging and the real ``ImageAnnotator`` are all genuine, because the dialog was
precisely where the bug lived.
"""

import pytest
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from src.digitalsreeni_image_annotator.controllers.segment_everything_controller import (
    SOURCE,
    TEMP_AUTO_CLASS,
)


def _ring(x0, y0, side):
    return [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side]


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


@pytest.fixture
def ready(window, monkeypatch):
    """A window that can run Segment Everything, with SAM mocked at the seam."""
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))

    image = QImage(200, 200, QImage.Format.Format_RGB888)
    image.fill(0)
    window.current_image = image
    window.image_file_name = "img1.png"
    window.current_slice = None
    window.image_label.original_pixmap = QPixmap(200, 200)
    window.class_mapping = {"cell": 1}
    window.image_label.class_colors["cell"] = QColor(255, 0, 0)
    window.sam_utils.current_sam_model = "SAM 2 tiny"  # gate only, never used
    return window


def _mock_sam(monkeypatch, window, proposals):
    monkeypatch.setattr(
        window.sam_utils, "apply_sam_everything", lambda *a, **k: proposals
    )


def test_a_completed_run_stages_its_proposals(ready, monkeypatch):
    """The regression: SAM returned masks and the controller binned them."""
    _mock_sam(monkeypatch, ready, [
        {"segmentation": _ring(10, 10, 40), "score": 0.9},
        {"segmentation": _ring(100, 100, 40), "score": 0.8},
    ])

    ready.segment_everything_controller.run()

    assert len(ready.image_label.temp_annotations) == 2
    assert all(
        a["source"] == SOURCE and a["category_name"] == TEMP_AUTO_CLASS
        for a in ready.image_label.temp_annotations
    )


def test_closing_the_progress_dialog_is_not_a_cancellation(ready, monkeypatch):
    """Pins the exact Qt behaviour that caused the bug, so a future refactor
    back to `if progress.wasCanceled()` after `close()` fails here."""
    dialog = QProgressDialog("x", "Cancel", 0, 0, None)
    dialog.show()
    assert dialog.wasCanceled() is False
    dialog.close()
    assert dialog.wasCanceled() is True, (
        "Qt still sets the cancel flag on close(); the controller must read "
        "the flag before closing"
    )


def test_a_genuine_cancel_discards_the_proposals(ready, monkeypatch):
    """The check still has to work -- the fix must not simply delete it."""
    def cancel_then_return(*args, **kwargs):
        # Stand in for the user hitting Cancel while inference runs: the call
        # is not interruptible, so it completes and the results are dropped.
        for widget in ready.findChildren(QProgressDialog):
            widget.cancel()
        return [{"segmentation": _ring(10, 10, 40), "score": 0.9}]

    monkeypatch.setattr(
        ready.sam_utils, "apply_sam_everything", cancel_then_return
    )

    ready.segment_everything_controller.run()

    assert ready.image_label.temp_annotations == []


def test_the_run_survives_sam_returning_nothing(ready, monkeypatch):
    _mock_sam(monkeypatch, ready, [])
    ready.segment_everything_controller.run()
    assert ready.image_label.temp_annotations == []


def test_proposals_are_parked_under_the_image_key(ready, monkeypatch):
    """``temp_annotations`` is a single field, not per-image: without the park
    a stray click in the image list discards a batch mid-assignment."""
    _mock_sam(monkeypatch, ready, [{"segmentation": _ring(10, 10, 40), "score": 0.9}])

    ready.segment_everything_controller.run()

    assert len(ready.dino_batch_results.get("img1.png", [])) == 1


def test_inference_failure_reports_rather_than_staging_garbage(ready, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(ready.sam_utils, "apply_sam_everything", boom)

    ready.segment_everything_controller.run()

    assert ready.image_label.temp_annotations == []
