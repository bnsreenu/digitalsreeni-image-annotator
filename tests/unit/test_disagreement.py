"""Model-vs-ground-truth disagreement and uncertainty scoring (issue #71).

Qt-free, like the QC rules, and for the same reason — but here the payoff is
different: the matching has adversarial cases (a greedy assignment that is
locally sensible and globally wrong) that would be painful to construct through
a real model and are trivial to construct as two lists of dicts.

The class-mapping test is the one guarding the most damaging silent failure:
predictions arrive as ``Temp-cell``, and without stripping that prefix every
prediction counts as unmatched, so *every* image scores badly and the ranking
is pure noise while looking entirely reasonable.
"""

import subprocess
import sys

import pytest

from src.digitalsreeni_image_annotator.core import disagreement as dg


def _square(x0, y0, side, name="cell", score=None):
    entry = {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
    }
    if score is not None:
        entry["score"] = score
    return entry


def _box(x, y, w, h, name="cell", score=None):
    entry = {"bbox": [x, y, w, h], "category_name": name}
    if score is not None:
        entry["score"] = score
    return entry


def _pose(name="person"):
    return {
        "keypoints": [10, 10, 2, 20, 20, 2],
        "num_keypoints": 2,
        "bbox": [10, 10, 10, 10],
        "category_name": name,
    }


# --- Qt-free guarantee -----------------------------------------------------


def test_the_scorer_imports_without_qt():
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.disagreement as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- class-name mapping ----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected", [("Temp-cell", "cell"), ("cell", "cell"), ("Temp-", "")]
)
def test_temp_prefix_is_stripped(raw, expected):
    assert dg.strip_temp_prefix(raw) == expected


def test_a_prefixed_prediction_still_matches_its_class():
    """Without the mapping every prediction is unmatched and every image scores
    badly — a ranking that is noise while looking entirely plausible."""
    gt = [_square(0, 0, 10, name="cell")]
    pred = [_square(0, 0, 10, name="Temp-cell", score=0.9)]
    score, breakdown = dg.disagreement_score(gt, pred)
    assert score == pytest.approx(0.0)
    assert breakdown["matched"] == 1


def test_a_prediction_of_the_wrong_class_is_a_disagreement():
    """Not a partial match: predicting 'nucleus' where the label says 'cell' is
    a miss and a spurious detection, not a well-fitting pair."""
    gt = [_square(0, 0, 10, name="cell")]
    pred = [_square(0, 0, 10, name="Temp-nucleus", score=0.9)]
    score, breakdown = dg.disagreement_score(gt, pred)
    assert breakdown["matched"] == 0
    assert score == pytest.approx(2.0)


# --- disagreement score ----------------------------------------------------


def test_perfect_agreement_scores_zero():
    annotations = [_square(0, 0, 10), _square(50, 50, 10)]
    predictions = [_square(0, 0, 10, score=0.9), _square(50, 50, 10, score=0.9)]
    score, _ = dg.disagreement_score(annotations, predictions)
    assert score == pytest.approx(0.0)


def test_a_missing_prediction_costs_one():
    score, breakdown = dg.disagreement_score([_square(0, 0, 10)], [])
    assert score == pytest.approx(1.0)
    assert breakdown["missed"] == 1


def test_a_spurious_prediction_costs_one():
    score, breakdown = dg.disagreement_score([], [_square(0, 0, 10, score=0.9)])
    assert score == pytest.approx(1.0)
    assert breakdown["spurious"] == 1


def test_a_poorly_fitting_pair_costs_its_shape_error():
    gt = [_square(0, 0, 10)]
    pred = [_square(3, 0, 10, score=0.9)]
    score, breakdown = dg.disagreement_score(gt, pred)
    assert breakdown["matched"] == 1
    assert 0 < score < 1, "a partial overlap is worse than perfect, better than absent"


def test_a_barely_overlapping_pair_is_two_errors_not_one_bad_match():
    """Below the match threshold the shapes are not the same object; pairing
    them would flatter the score with a meaningless near-miss."""
    gt = [_square(0, 0, 10)]
    pred = [_square(9, 0, 10, score=0.9)]
    score, breakdown = dg.disagreement_score(gt, pred)
    assert breakdown["matched"] == 0
    assert score == pytest.approx(2.0)


def test_bbox_and_polygon_annotations_compare():
    gt = [_square(0, 0, 10)]
    pred = [_box(0, 0, 10, 10, name="Temp-cell", score=0.9)]
    score, breakdown = dg.disagreement_score(gt, pred)
    assert breakdown["matched"] == 1
    assert score == pytest.approx(0.0, abs=0.01)


def test_empty_inputs_score_zero():
    score, breakdown = dg.disagreement_score([], [])
    assert score == 0.0
    assert breakdown["matched"] == 0
    assert dg.disagreement_score(None, None)[0] == 0.0


# --- matching --------------------------------------------------------------


def test_greedy_and_optimal_differ_and_optimal_wins():
    """The case plain greedy gets wrong.

    gt0 overlaps pred0 best, so greedy takes that pair first and leaves gt1
    with whatever remains. Swapping partners raises the total IoU, and the
    swap pass has to find it.
    """
    gt = [_square(0, 0, 10), _square(6, 0, 10)]
    pred = [_square(1, 0, 10, name="Temp-cell"), _square(6, 0, 10, name="Temp-cell")]

    pairs, unmatched_gt, unmatched_pred = dg.match_pairs(gt, pred)

    assert len(pairs) == 2
    assert unmatched_gt == [] and unmatched_pred == []
    total = sum(value for _i, _j, value in pairs)
    # The assignment gt0->pred0, gt1->pred1 is the optimum here; any other
    # pairing scores strictly lower.
    assert dict((i, j) for i, j, _v in pairs) == {0: 0, 1: 1}
    assert total > 1.5


def test_matching_never_reuses_a_prediction():
    gt = [_square(0, 0, 10), _square(1, 0, 10)]
    pred = [_square(0, 0, 10, name="Temp-cell")]
    pairs, unmatched_gt, _ = dg.match_pairs(gt, pred)
    assert len(pairs) == 1
    assert len(unmatched_gt) == 1


def test_matching_is_per_class():
    gt = [_square(0, 0, 10, name="cell"), _square(0, 0, 10, name="nucleus")]
    pred = [_square(0, 0, 10, name="Temp-nucleus")]
    pairs, _unmatched_gt, _ = dg.match_pairs(gt, pred)
    assert [p[0] for p in pairs] == [1], "paired with the nucleus, not the cell"


# --- pose exclusion --------------------------------------------------------


def test_pose_instances_are_not_scorable():
    assert dg.is_scorable(_pose()) is False
    assert dg.is_scorable(_square(0, 0, 10)) is True
    assert dg.is_scorable(_box(0, 0, 5, 5)) is True


def test_pose_instances_are_excluded_and_reported():
    """Silently scoring a pose project as perfect would be worse than not
    scoring it; the count is surfaced so the UI can say so."""
    score, breakdown = dg.disagreement_score([_pose(), _square(0, 0, 10)],
                                             [_square(0, 0, 10, name="Temp-cell")])
    assert breakdown["skipped_pose"] == 1
    assert score == pytest.approx(0.0)


# --- uncertainty -----------------------------------------------------------


def test_a_borderline_detection_is_maximally_uncertain():
    score, _ = dg.uncertainty_score([{"score": 0.5}])
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_a_confident_detection_contributes_nothing(confidence):
    score, _ = dg.uncertainty_score([{"score": confidence}])
    assert score == pytest.approx(0.0)


def test_uncertainty_sums_rather_than_averages():
    """Ten borderline detections teach more than one; averaging would hide it."""
    one, _ = dg.uncertainty_score([{"score": 0.5}])
    ten, _ = dg.uncertainty_score([{"score": 0.5}] * 10)
    assert ten == pytest.approx(one * 10)


def test_an_image_with_no_detections_scores_zero():
    """The model seeing nothing is not the model being unsure; conflating them
    floods the top of the ranking with empty images."""
    score, breakdown = dg.uncertainty_score([])
    assert score == 0.0
    assert breakdown["detections"] == 0


def test_tracked_detections_are_excluded_from_uncertainty():
    """SAM 3 tracking writes a constant confidence of 1.0 (ADR-040), so its
    'certainty' carries no information at all."""
    score, breakdown = dg.uncertainty_score(
        [{"score": 0.5, "source": "sam3-track"}, {"score": 0.5, "source": "dino"}]
    )
    assert breakdown["detections"] == 1
    assert score == pytest.approx(1.0)


def test_detections_without_a_confidence_are_skipped():
    score, breakdown = dg.uncertainty_score([{"category_name": "cell"}])
    assert score == 0.0
    assert breakdown["detections"] == 0


# --- ranking ---------------------------------------------------------------


def test_rank_orders_worst_first():
    ranked = dg.rank({"a.png": 1.0, "b.png": 5.0, "c.png": 3.0})
    assert [name for name, _ in ranked] == ["b.png", "c.png", "a.png"]


def test_rank_breaks_ties_by_name_for_stability():
    """The ranking is something a person works down over several sessions; an
    unstable order would reshuffle it under them."""
    ranked = dg.rank({"b.png": 2.0, "a.png": 2.0, "c.png": 2.0})
    assert [name for name, _ in ranked] == ["a.png", "b.png", "c.png"]


def test_rank_handles_an_empty_mapping():
    assert dg.rank({}) == []
    assert dg.rank(None) == []
