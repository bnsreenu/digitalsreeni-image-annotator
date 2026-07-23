"""SAMTrainController UI-locking tests (issue bnsreenu#73).

The lock/unlock of the SAM inference UI during a fine-tuning run is exactly the
kind of state machine that regresses silently. These build one real
ImageAnnotator (offscreen) and exercise the controller directly — no model
weights, no worker thread.
"""

import pytest


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def test_set_sam_ui_locked_toggles_widgets(window):
    # The helper's contract is the toggle itself; the buttons' construction-time
    # enabled state depends on whether an image is loaded, so assert relative
    # to an explicit unlocked baseline rather than the initial state.
    c = window.sam_train_controller
    widgets = [window.sam_box_button, window.sam_points_button, window.sam_model_selector]

    c._set_sam_ui_locked(True)
    assert not any(w.isEnabled() for w in widgets)
    assert not c._menu.isEnabled()

    c._set_sam_ui_locked(False)
    assert all(w.isEnabled() for w in widgets)
    assert c._menu.isEnabled()


def test_launch_unlocks_ui_when_setup_raises(window, monkeypatch):
    """If anything between locking and thread.start() raises, the SAM UI must
    be restored — otherwise the tools stay dead until app restart."""
    from PyQt6.QtWidgets import QMessageBox

    import digitalsreeni_image_annotator.controllers.sam_train_controller as mod
    from digitalsreeni_image_annotator.dialogs.sam_trainer_dialog import (
        SAMTrainConfigDialog,
    )

    c = window.sam_train_controller
    monkeypatch.setattr(c, "_gpu_gate", lambda: True)
    monkeypatch.setattr(
        SAMTrainConfigDialog, "exec",
        lambda self: SAMTrainConfigDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        SAMTrainConfigDialog, "get_config",
        lambda self: {
            "base_model": "SAM 2 tiny", "out_name": "t", "epochs": 1,
            "lr": 1e-4, "batch_size": 1, "prompt_type": "bbox",
            "freeze_image_encoder": True,
        },
    )

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("setup failed on purpose")

    monkeypatch.setattr(mod, "SAMFineTuner", Boom)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    c._launch([object()])  # group content irrelevant — fails before use

    assert window.sam_box_button.isEnabled()
    assert window.sam_points_button.isEnabled()
    assert window.sam_model_selector.isEnabled()
    assert c._menu.isEnabled()


def test_launch_always_wires_a_real_mlflow_tracker(window, monkeypatch):
    """The always-on guarantee (ADR-027): the GUI path always hands the trainer
    a real MLflowTracker — there is no off switch and never a None/_NullTracker.
    Guards against a silent revert of the controller's tracker wiring."""
    import digitalsreeni_image_annotator.controllers.sam_train_controller as mod
    from digitalsreeni_image_annotator.dialogs.sam_trainer_dialog import (
        SAMTrainConfigDialog,
    )
    from digitalsreeni_image_annotator.training.mlflow_tracker import MLflowTracker

    c = window.sam_train_controller
    monkeypatch.setattr(c, "_gpu_gate", lambda: True)
    monkeypatch.setattr(
        SAMTrainConfigDialog, "exec",
        lambda self: SAMTrainConfigDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        SAMTrainConfigDialog, "get_config",
        lambda self: {
            "base_model": "SAM 2 tiny", "out_name": "t", "epochs": 1,
            "lr": 1e-4, "batch_size": 1, "prompt_type": "bbox",
            "freeze_image_encoder": True,
        },
    )

    # Capture the cfg handed to the worker thread without starting one.
    captured = {}

    class _Sig:
        def connect(self, *a, **k):
            pass

    class FakeThread:
        finished = _Sig()

        def __init__(self, finetuner, base_model, groups, cfg):
            captured["cfg"] = cfg

        def start(self):
            pass

    monkeypatch.setattr(mod, "SAMTrainingThread", FakeThread)

    c._launch([object()])

    tracker = captured["cfg"]["tracker"]
    assert isinstance(tracker, MLflowTracker)  # real tracker, never None/_NullTracker
    assert tracker._run_name == "t"  # wired with the run name from the dialog


def test_mlflow_run_url_shows_link_and_opens_browser(window, monkeypatch):
    """When a run opens, the progress dialog shows a clickable deep link, the
    MLflow UI server is started once, and the browser opens for each run."""
    import webbrowser

    from PyQt6.QtCore import QTimer

    import digitalsreeni_image_annotator.training.mlflow_tracker as mt
    from digitalsreeni_image_annotator.dialogs.yolo_trainer import (
        TrainingInfoDialog,
    )

    c = window.sam_train_controller
    window.sam_training_dialog = TrainingInfoDialog(window)

    started = []
    monkeypatch.setattr(
        mt, "start_mlflow_ui_server",
        lambda uri, log=None: (started.append(uri), (True, "ok"))[1],
    )
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u: opened.append(u))
    # Fire the deferred browser-open immediately instead of after the delay.
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda ms, fn: fn()))

    url = "http://localhost:5000/#/experiments/1/runs/abc"
    c._on_mlflow_run_url(url)
    c._on_mlflow_run_url(url)  # a second run must NOT restart the server

    assert url in window.sam_training_dialog.info_text.toHtml()  # clickable link
    assert len(started) == 1  # server launched once (guarded)
    assert opened == [url, url]  # browser opened for each run


def test_link_format_does_not_bleed_into_later_lines(qt_application):
    """A clickable run link must not turn the plain progress lines appended
    after it into links too (regression: epoch rows rendered as anchors)."""
    from digitalsreeni_image_annotator.dialogs.yolo_trainer import (
        TrainingInfoDialog,
    )

    dlg = TrainingInfoDialog()
    url = "http://localhost:5000/#/experiments/1/runs/abc"
    dlg.update_info("Epoch 1/10 loss=0.02")
    dlg.update_info_link("Open this run in MLflow", url)
    dlg.update_info("Epoch 2/10 loss=0.01")  # used to inherit the anchor format
    dlg.update_info("Epoch 3/10 loss=0.01")

    html = dlg.info_text.toHtml()
    assert url in html  # the link is present and clickable
    assert html.count("href=") == 1  # exactly one anchor — only the link line
    dlg.deleteLater()
