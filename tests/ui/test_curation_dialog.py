"""The dataset similarity report's new controls (#82).

The dialog is thin, but three things in it are easy to get wrong in ways that
look fine: a failed backend switch leaving the report empty, an uncertainty
column that is blank because nothing measured it rather than because nothing is
uncertain, and a threshold slider that fires a full pass per pixel of drag.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QListWidget, QWidget

from src.digitalsreeni_image_annotator.controllers.curation_controller import (
    CurationController,
)
from src.digitalsreeni_image_annotator.controllers.review_controller import (
    MODE_UNCERTAINTY,
)
from src.digitalsreeni_image_annotator.dialogs.dataset_curation_dialog import (
    DatasetCurationDialog,
)


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.all_images = []
        self.image_paths = {}
        self.image_slices = {}
        self.image_list = QListWidget()
        self.current_project_file = None


class _Review:
    def __init__(self, scores):
        self.scores = scores

    def has_scores(self):
        return bool(self.scores)

    def score_for(self, name):
        record = self.scores.get(name)
        return record["score"] if record else None

    def mode_for(self, name):
        record = self.scores.get(name)
        return record["mode"] if record else None


def _unit(x, y):
    vector = np.array([x, y], dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def controller(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    made = CurationController(window)
    made.embeddings = {
        "a1.png": _unit(1.0, 0.00),
        "a2.png": _unit(1.0, 0.01),
        "a3.png": _unit(1.0, 0.02),
        "lonely.png": _unit(0.0, 1.00),
    }
    return made


@pytest.fixture
def dialog(qtbot, controller):
    made = DatasetCurationDialog(controller.mw, controller)
    qtbot.addWidget(made)
    return made


# --- the report ------------------------------------------------------------


def test_a_cluster_reports_its_cohesion(dialog):
    """Without it, a chain of frames and a tight burst look identical."""
    parent = dialog.tree.topLevelItem(0)
    assert parent.text(0) == "Cluster 1"
    assert "/" in parent.text(2), "no cohesion shown for a cluster"


def test_the_coverage_line_names_the_threshold_it_used(dialog):
    """The mode count is a coarse heuristic and model-dependent. Stated without
    its threshold it would read as ground truth."""
    text = dialog.coverage_label.text()
    assert "appearance mode" in text
    assert f"{dialog.controller.mode_threshold:.2f}" in text


def test_isolated_images_are_still_listed(dialog):
    labels = [
        dialog.tree.topLevelItem(row).text(0)
        for row in range(dialog.tree.topLevelItemCount())
    ]
    assert "Isolated images" in labels


# --- the review-score column ----------------------------------------------


def test_the_review_column_is_hidden_when_nothing_measured_it(dialog):
    """An empty column reads as "no uncertainty here" rather than "no review
    has run" -- and on a video project a review never can run."""
    assert dialog.tree.isColumnHidden(3)


def test_review_scores_switch_the_suggestion_to_the_most_uncertain(
    qtbot, controller
):
    controller.mw.review_controller = _Review({
        "a1.png": {"score": 0.1, "mode": MODE_UNCERTAINTY},
        "a2.png": {"score": 0.2, "mode": MODE_UNCERTAINTY},
        "a3.png": {"score": 0.9, "mode": MODE_UNCERTAINTY},
    })
    made = DatasetCurationDialog(controller.mw, controller)
    qtbot.addWidget(made)

    assert not made.tree.isColumnHidden(3)
    assert made.tree.topLevelItem(0).text(4) == "a3.png (most uncertain)"


# --- the backend picker ----------------------------------------------------


def test_the_picker_offers_every_model_the_controller_accepts(dialog):
    offered = [
        dialog.model_combo.itemText(index)
        for index in range(dialog.model_combo.count())
    ]
    assert offered == dialog.controller.available_models()
    assert dialog.model_combo.currentText() == dialog.controller.model_name


def test_switching_backend_recomputes_and_refreshes(dialog, monkeypatch):
    replacement = {
        "a1.png": _unit(1.0, 0.0),
        "a2.png": _unit(0.0, 1.0),
    }

    def _compute(_parent=None):
        dialog.controller.embeddings = replacement
        return True

    monkeypatch.setattr(dialog.controller, "compute", _compute)
    dialog.model_combo.setCurrentText("DINOv2 (base)")

    assert dialog.controller.model_name == "DINOv2 (base)"
    assert dialog.controller.embeddings is replacement
    # The report is rebuilt from the new vectors, not left showing the old.
    assert "of 2 images" in dialog.summary_label.text()


def test_a_failed_switch_puts_the_previous_model_back(dialog, monkeypatch):
    """No network for the download, or the user cancelling the progress dialog.
    Leaving the report empty and the combo pointing at a model with no vectors
    would be a worse outcome than not switching."""
    before_model = dialog.controller.model_name
    before_embeddings = dialog.controller.embeddings

    monkeypatch.setattr(dialog.controller, "compute", lambda _parent=None: False)
    dialog.model_combo.setCurrentText("DINOv2 (base)")

    assert dialog.controller.model_name == before_model
    assert dialog.controller.embeddings is before_embeddings
    assert dialog.model_combo.currentText() == before_model
    assert dialog.tree.topLevelItemCount() > 0, "the report was left empty"


def test_the_picker_is_inert_while_a_run_is_in_flight(dialog, monkeypatch):
    """`compute` spins the event loop on every item and its progress dialog is
    non-modal, so the combo stays clickable for the whole run. A second
    selection used to unload the model the outer loop was still using and leave
    a mixed CLIP+DINOv2 embedding set behind — both are 768-d, so nothing
    downstream can detect it, and `refine` feeds those clusters into a real
    training run's split."""
    monkeypatch.setattr(dialog.controller, "is_computing", lambda: True)
    before = dialog.controller.model_name
    monkeypatch.setattr(
        dialog.controller,
        "compute",
        lambda _parent=None: pytest.fail("re-entered a run in flight"),
    )

    dialog.model_combo.setCurrentText("DINOv2 (base)")

    assert dialog.controller.model_name == before
    assert dialog.model_combo.currentText() == before


def test_the_controls_are_disabled_for_the_duration_of_a_run(dialog, monkeypatch):
    """`is_computing` proves the branch exists; this proves it is reachable.

    Disabling is what stops a *user* re-entering, which is the actual vector:
    `compute` spins `processEvents` behind a non-modal dialog, so clicks are
    delivered mid-run.
    """
    states = []

    def _compute(_parent=None):
        states.append(
            (dialog.model_combo.isEnabled(), dialog.slider.isEnabled())
        )
        return True

    monkeypatch.setattr(dialog.controller, "compute", _compute)
    dialog.model_combo.setCurrentText("DINOv2 (base)")

    assert states == [(False, False)], "the controls stayed live during the run"
    assert dialog.model_combo.isEnabled() and dialog.slider.isEnabled()


def test_isolated_images_carry_their_review_score(qtbot, controller):
    """Gathering scores from cluster members alone hid the column outright on a
    project where only the isolated images were scored -- and those are the
    ones whose uncertainty matters most."""
    controller.mw.review_controller = _Review({
        "lonely.png": {"score": 7.5, "mode": MODE_UNCERTAINTY},
    })
    made = DatasetCurationDialog(controller.mw, controller)
    qtbot.addWidget(made)

    assert not made.tree.isColumnHidden(3)
    isolated = [
        made.tree.topLevelItem(row)
        for row in range(made.tree.topLevelItemCount())
        if made.tree.topLevelItem(row).text(0) == "Isolated images"
    ]
    assert isolated, "no isolated group in the report"
    assert isolated[0].child(0).text(3) == "7.5"


# --- the threshold slider --------------------------------------------------


def test_dragging_the_slider_does_not_re_analyse_per_tick(dialog, monkeypatch):
    """Each pass is milliseconds on a small project and seconds at the ceiling;
    a drag would queue up dozens of them."""
    calls = []
    monkeypatch.setattr(
        dialog.controller, "analyse", lambda *a, **k: calls.append(1) or {
            "clusters": [], "outliers": [], "modes": [], "mode_threshold": 0.8
        }
    )

    for value in range(90, 96):
        dialog.slider.setValue(value)

    assert calls == []
    assert dialog.threshold_label.text() == "0.95", "the label must track live"


def test_the_slider_re_analyses_once_the_drag_settles(qtbot, dialog, monkeypatch):
    calls = []
    monkeypatch.setattr(
        dialog.controller, "analyse", lambda *a, **k: calls.append(1) or {
            "clusters": [], "outliers": [], "modes": [], "mode_threshold": 0.8
        }
    )

    dialog.slider.setValue(97)
    qtbot.waitUntil(lambda: bool(calls), timeout=2000)

    assert dialog.controller.threshold == pytest.approx(0.97)
