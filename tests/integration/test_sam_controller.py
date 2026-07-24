"""SAMController debounce + in-flight guard tests (fork issue #36).

``controllers/sam_controller.py`` implements ADR-013's debounce timer and the
in-flight re-entrancy guard — a state machine that "regresses silently". These
build one real ImageAnnotator (offscreen) and exercise the controller
directly, monkeypatching methods on the real ``window.sam_utils`` instance so
**no model is ever constructed, no weights downloaded, and the 1s debounce
timer is never fired** (its state is asserted instead — firing it would enter
real inference).
"""

import pytest

from PyQt6.QtWidgets import QMessageBox

from digitalsreeni_image_annotator.inference.sam_utils import InferenceBusyError


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _arm_sam_points(window, points):
    """Put the canvas into the sam_points tool with the given positive points."""
    window.image_label.current_tool = "sam_points"
    window.image_label.sam_positive_points = list(points)
    window.image_label.sam_negative_points = []
    window.image_label.temp_sam_prediction = None


# ── debounce timer state (never fired) ──────────────────────────────────────

def test_schedule_starts_1s_singleshot_debounce(window):
    window.schedule_sam_prediction()

    timer = window.sam_inference_timer
    assert timer.isActive()
    assert timer.isSingleShot()
    assert timer.interval() == 1000

    # Restart semantics: scheduling again keeps it active.
    window.schedule_sam_prediction()
    assert timer.isActive()
    assert timer.interval() == 1000


def test_cancel_stops_debounce(window):
    window.schedule_sam_prediction()
    assert window.sam_inference_timer.isActive()

    window.sam_controller.cancel_sam_debounce()
    assert not window.sam_inference_timer.isActive()


# ── apply_sam_prediction: guard / error / success paths ─────────────────────

def test_in_flight_guard_blocks_reentry(window):
    """The ADR-013 guard bails BEFORE the try/finally: the recorder is never
    called and the flag stays True (only the owning call clears it)."""
    called = []
    window.sam_utils.apply_sam_points = lambda *a, **k: called.append((a, k))

    window._sam_inference_in_flight = True
    _arm_sam_points(window, [(1, 1)])

    window.apply_sam_prediction()

    assert called == []  # guard returned before touching sam_utils
    assert window._sam_inference_in_flight is True  # guard doesn't clear it


def test_busy_error_swallowed_silently(window, monkeypatch):
    """``InferenceBusyError`` from ``sam_utils`` is caught silently — no dialog,
    no exception escapes, and the flag is cleared by ``finally``."""
    def _raise_busy(*a, **k):
        raise InferenceBusyError("busy")

    window.sam_utils.apply_sam_points = _raise_busy

    def _no_dialog(*a, **k):
        raise AssertionError("QMessageBox.critical must not be called")

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_no_dialog))

    _arm_sam_points(window, [(1, 1)])
    assert window._sam_inference_in_flight is False

    window.apply_sam_prediction()  # must not raise

    assert window._sam_inference_in_flight is False  # finally cleared it


def test_error_shows_dialog_and_clears_flag(window, monkeypatch):
    """A generic error dialogs exactly once (text carries the exception) and
    the flag is still cleared by ``finally``."""
    def _raise_boom(*a, **k):
        raise RuntimeError("boom")

    window.sam_utils.apply_sam_points = _raise_boom

    captured = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        staticmethod(lambda *a, **k: captured.append(a)),
    )

    _arm_sam_points(window, [(1, 1)])

    window.apply_sam_prediction()

    assert len(captured) == 1
    # args are (parent, title, text) — the exception text is embedded.
    assert "boom" in captured[0][2]
    assert window._sam_inference_in_flight is False


def test_success_sets_temp_prediction(window, monkeypatch):
    """A truthy prediction populates ``temp_sam_prediction`` with the current
    class name and the returned score; the flag is cleared afterwards."""
    monkeypatch.setattr(window, "auto_save", lambda *a, **k: None)
    window.add_class("cell")

    window.sam_utils.apply_sam_points = lambda *a, **k: {
        "segmentation": [1, 1, 5, 1, 5, 5],
        "score": 0.9,
    }

    _arm_sam_points(window, [(1, 1)])

    window.apply_sam_prediction()

    temp = window.image_label.temp_sam_prediction
    assert temp is not None
    assert temp["category_name"] == "cell"
    assert temp["score"] == 0.9
    assert window._sam_inference_in_flight is False


def test_no_positive_points_noop(window, monkeypatch):
    """sam_points with no positive points returns before calling sam_utils and
    without any dialog."""
    called = []
    window.sam_utils.apply_sam_points = lambda *a, **k: called.append((a, k))

    dialogs = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: dialogs.append(a))
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: dialogs.append(a))
    )

    _arm_sam_points(window, [])

    window.apply_sam_prediction()

    assert called == []
    assert dialogs == []
    assert window._sam_inference_in_flight is False


# ── change_sam_model: failure/success plumbing ──────────────────────────────

def test_change_model_failure_resets_selector(window, monkeypatch):
    """A load failure dialogs, resets the selector to index 0, and leaves
    ``current_sam_model`` untouched (still None)."""
    def _raise(model_name):
        # The controller resets the selector to index 0 on failure, which fires
        # the connected slot with "Pick a SAM Model" (the unset path). Real
        # sam_utils never raises for that, so neither does the stub — only the
        # actual model load fails, yielding exactly one dialog.
        if model_name == "Pick a SAM Model":
            window.sam_utils.current_sam_model = None
            return
        raise RuntimeError("no weights")

    window.sam_utils.change_sam_model = _raise

    captured = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: captured.append(a))
    )

    # Move the selector off index 0 without firing the connected slot.
    window.sam_model_selector.blockSignals(True)
    window.sam_model_selector.setCurrentIndex(1)
    window.sam_model_selector.blockSignals(False)
    assert window.sam_model_selector.currentIndex() == 1

    window.sam_controller.change_sam_model("SAM 2 tiny")

    assert len(captured) == 1
    assert window.sam_model_selector.currentIndex() == 0
    assert window.current_sam_model is None  # unchanged on failure


def test_change_model_success_syncs_name(window, monkeypatch):
    """On success ``mw.current_sam_model`` is synced from ``sam_utils`` — with
    the CPU-fallback warning (which can open a dialog) stubbed out."""
    def _load(model_name):
        window.sam_utils.current_sam_model = "SAM 2 tiny"

    window.sam_utils.change_sam_model = _load

    import digitalsreeni_image_annotator.core.torch_utils as torch_utils
    monkeypatch.setattr(torch_utils, "maybe_warn_cpu_fallback", lambda *a, **k: None)

    window.sam_controller.change_sam_model("SAM 2 tiny")

    assert window.current_sam_model == "SAM 2 tiny"
