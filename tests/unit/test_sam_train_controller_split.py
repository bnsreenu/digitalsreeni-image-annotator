"""The SAM launch path hands the split its exact grouping (#82, ADR-045).

`SAMFineTuner.train` splits on the training thread. If it re-derived the
grouping from `SampleGroup.name` there, the curation refinement the user was
*warned about* would be silently dropped and the run would perform a different
split than the dialog described — the same divergence class ADR-044 spent two
revisions closing.

A senior review found this wiring had no coverage at all: deleting both the
refinement call and the config assignment left the whole suite green.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QListWidget, QWidget

from src.digitalsreeni_image_annotator.controllers.curation_controller import (
    CurationController,
)
from src.digitalsreeni_image_annotator.controllers.sam_train_controller import (
    SAMTrainController,
)
from src.digitalsreeni_image_annotator.training.sam_dataset import split_keys
from src.digitalsreeni_image_annotator.training.sam_trainer import SampleGroup


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.all_images = []
        self.image_paths = {}
        self.image_slices = {}
        self.image_list = QListWidget()
        self.current_project_file = None


def _groups():
    return [
        SampleGroup(lambda: None, [{"bbox": [0, 0, 1, 1]}], name=name)
        for name in ("burst_a.png", "burst_b.png", "other.png")
    ]


@pytest.fixture
def controller(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    window.curation_controller = CurationController(window)
    made = SAMTrainController.__new__(SAMTrainController)
    made.mw = window
    return made


def test_the_config_carries_the_grouping_the_warning_describes(controller):
    keyed, config = controller._training_config({"train_pct": 80}, _groups())

    assert "keyed_groups" in config, "the worker will re-derive the grouping"
    assert set(config["keyed_groups"]) == set(keyed)
    # The dialog's own keys survive alongside it.
    assert config["train_pct"] == 80


def test_near_duplicate_clusters_reach_the_keyed_grouping(controller):
    """These three files are independent by name. Only the clusters can link
    two of them — and only if they are translated from names to split keys
    first, which is the step that fails silently when it is missed."""
    controller.mw.curation_controller.embeddings = {
        "burst_a.png": np.array([1.0, 0.0], dtype=np.float32),
        "burst_b.png": np.array([1.0, 0.01], dtype=np.float32),
        "other.png": np.array([0.0, 1.0], dtype=np.float32),
    }
    groups = _groups()

    _keyed, config = controller._training_config({"train_pct": 80}, groups)
    refined = config["keyed_groups"]
    _, derived = split_keys(groups)

    assert refined != derived, "the curation refinement never arrived"
    assert refined["0:burst_a.png"] == refined["1:burst_b.png"]
    assert refined["2:other.png"] != refined["0:burst_a.png"]


def test_without_a_curation_run_the_grouping_is_the_derived_one(controller):
    groups = _groups()
    _keyed, config = controller._training_config({"train_pct": 80}, groups)
    assert config["keyed_groups"] == split_keys(groups)[1]


def test_the_dialog_config_is_not_mutated(controller):
    """`_launch` pops keys out of the returned config.

    Today that is harmless -- `SAMTrainConfigDialog.get_config` returns a fresh
    dict literal each call -- so this pins the copy rather than fixing a live
    bug. Stated plainly because an earlier version of this docstring invented a
    failure scenario that could not happen.
    """
    original = {"train_pct": 80}
    controller._training_config(original, _groups())
    assert original == {"train_pct": 80}
