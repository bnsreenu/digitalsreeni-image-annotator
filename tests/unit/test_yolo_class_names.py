"""Prediction-time state on YOLOTrainer (crashes found in manual use).

``self.model`` has three producers -- ``load_model``, ``load_prediction_model``
and a finished training run -- but ``class_names`` and
``prediction_keypoint_schema`` were populated by only one of them. A model
trained in-app is the active prediction model with no separate load step, so
both were still None and the first prediction did ``class_names[class_id]``:
"'NoneType' object is not subscriptable", reported as a probable mismatch
between the model and the YAML file classes -- the one explanation that was
certainly wrong, since no YAML was involved.

Three rules, one per producer: the trained run sets them, a load replaces them,
and ``class_name_for`` falls back to the model's own names, which are always
there.
"""

import pytest

from src.digitalsreeni_image_annotator.dialogs.yolo_trainer import YOLOTrainer


def _trainer(class_names=None, model_names=None):
    trainer = YOLOTrainer.__new__(YOLOTrainer)
    trainer.class_names = class_names
    trainer.model = type("M", (), {"names": model_names})() if model_names else None
    return trainer


def test_a_freshly_trained_model_resolves_from_its_own_names():
    """The crash: nothing loaded, so class_names is None."""
    trainer = _trainer(class_names=None, model_names={0: "bee"})
    assert trainer.class_name_for(0) == "bee"


def test_loaded_names_win_over_the_models_own():
    trainer = _trainer(class_names={0: "from-yaml"}, model_names={0: "from-model"})
    assert trainer.class_name_for(0) == "from-yaml"


def test_a_list_of_names_works_too():
    """A hand-written yaml may carry `names: [bee]` rather than a mapping."""
    assert _trainer(class_names=["bee", "wasp"]).class_name_for(1) == "wasp"


def test_an_unknown_index_raises_IndexError_not_KeyError():
    """Both callers catch IndexError to report a genuine class mismatch; a
    dict's native KeyError would sail past them and reach the user as an
    unhandled crash instead."""
    trainer = _trainer(class_names={0: "bee"})
    with pytest.raises(IndexError):
        trainer.class_name_for(7)


def test_no_names_anywhere_raises_rather_than_returning_none():
    trainer = _trainer(class_names=None, model_names=None)
    with pytest.raises(IndexError):
        trainer.class_name_for(0)


# --- loading a different model must not inherit the last one's state --------


def test_load_model_drops_the_previous_models_prediction_state(tmp_path, monkeypatch):
    """Train a run, then load some other checkpoint, then predict.

    Before the fallback existed this sequence raised the loud NoneType error.
    With it, a stale ``class_names`` would instead have the new model's class 0
    confidently reported under the *old* run's name -- mislabelled temp
    annotations the user can accept and export. Loading has to clear.
    """
    import ultralytics

    trainer = YOLOTrainer.__new__(YOLOTrainer)
    trainer.class_names = {0: "bee"}
    trainer.prediction_keypoint_schema = {"names": ["a"], "skeleton": [], "flip_idx": [0]}
    trainer.model = None
    trainer.loaded_model_path = None

    other = tmp_path / "other.pt"
    other.write_bytes(b"fake")
    monkeypatch.setattr(
        ultralytics, "YOLO", lambda path: type("M", (), {"names": {0: "person"}})()
    )

    assert trainer.load_model(str(other)) is True

    assert trainer.class_names is None
    assert trainer.prediction_keypoint_schema is None
    assert trainer.class_name_for(0) == "person", "inherited the old run's names"
