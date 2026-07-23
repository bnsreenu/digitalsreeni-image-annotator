"""
Unit tests for the SAM fine-tuner's patience-based early stopping (issue #85).

`EarlyStopper` tracks the best (lowest) value seen and the number of epochs
since the last improvement, and stops once that count reaches ``patience``
(``patience == 0`` disables stopping). The best epoch is always tracked so the
trainer can save the best checkpoint rather than the last.
"""

from src.digitalsreeni_image_annotator.training.early_stop import EarlyStopper


def test_improvement_resets_and_tracks_best():
    es = EarlyStopper(patience=3)
    assert es.update(1.0, 1) is True
    assert es.update(0.5, 2) is True
    assert es.best == 0.5 and es.best_epoch == 2
    assert es.update(0.6, 3) is False
    assert es.num_bad == 1


def test_stops_after_patience_bad_epochs():
    es = EarlyStopper(patience=2)
    es.update(1.0, 1)
    es.update(0.9, 2)        # improves -> resets
    assert not es.should_stop
    es.update(1.0, 3)        # bad #1
    assert not es.should_stop
    es.update(1.0, 4)        # bad #2 -> patience reached
    assert es.should_stop


def test_patience_zero_never_stops():
    es = EarlyStopper(patience=0)
    for e in range(1, 50):
        es.update(float(e), e)  # always worse
    assert not es.should_stop
    assert es.best == 1.0 and es.best_epoch == 1  # best still tracked


def test_min_delta_requires_real_improvement():
    es = EarlyStopper(patience=5, min_delta=0.1)
    es.update(1.0, 1)
    assert es.update(0.95, 2) is False  # within min_delta -> not an improvement
    assert es.update(0.85, 3) is True
