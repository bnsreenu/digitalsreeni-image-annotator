"""Review scoring against the shape ``YOLOTrainer.predict`` actually returns.

Written after end-to-end verification with a real yolo11n.pt caught a silent
P0: ``predict`` returns ``(results, input_size, original_size)`` —
``process_yolo_results`` unpacks that explicitly, and scoring did not. Iterating
the triple yielded three objects with no ``.boxes``, so **every image produced
zero predictions**. Nothing raised. Annotated images simply scored "every label
missed", unannotated ones scored 0, and the resulting ranking looked entirely
plausible while meaning nothing — the exact failure the disagreement module's
docstring warns about.

The unit tests that existed passed a bare list, which is why they missed it.
These pass the real return shape.
"""

import pytest
from PyQt6.QtWidgets import QWidget

from src.digitalsreeni_image_annotator.controllers.review_controller import (
    MODE_DISAGREEMENT,
    MODE_UNCERTAINTY,
    ReviewController,
    _unwrap_results,
)


class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = [xyxy]


class _Boxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Result:
    def __init__(self, boxes, names):
        self.boxes = _Boxes(boxes)
        self.names = names
        self.masks = None


def _results():
    return [_Result(
        [_Box(0, 0.9, [10, 10, 60, 60]), _Box(1, 0.5, [100, 100, 160, 160])],
        {0: "cell", 1: "nucleus"},
    )]


TRIPLE = None  # built per test so the results object is never shared


class _Trainer:
    """Returns exactly what YOLOTrainer.predict returns."""

    model = object()

    def predict(self, _path):
        return _results(), (640, 480), (640, 480)


class _Window(QWidget):
    def __init__(self, annotations=None):
        super().__init__()
        self.all_annotations = annotations or {}
        self.all_images = []
        self.image_paths = {}
        self.image_file_name = ""


@pytest.fixture
def controller(qtbot):
    win = _Window()
    qtbot.addWidget(win)
    return ReviewController(win)


# --- the unwrapper ---------------------------------------------------------


def test_the_predict_triple_is_unwrapped():
    results = _results()
    assert _unwrap_results((results, (640, 480), (640, 480))) is results


def test_a_bare_results_list_passes_through():
    results = _results()
    assert _unwrap_results(results) is results


def test_none_is_safe():
    assert _unwrap_results(None) == []


def test_an_unrelated_three_tuple_is_not_mistaken_for_the_triple():
    """The shape test checks the trailing size tuples, so three Results in a
    tuple are still iterated as results."""
    a, b, c = _Result([], {}), _Result([], {}), _Result([], {})
    assert _unwrap_results((a, b, c)) == (a, b, c)


# --- extraction ------------------------------------------------------------


def test_extraction_from_the_real_predict_shape(controller):
    """THE regression. Passing the triple used to yield zero predictions."""
    preds = controller.extract_predictions(
        (_results(), (640, 480), (640, 480)), "a.png"
    )
    assert len(preds) == 2, "the predict triple produced no predictions"
    assert preds[0]["category_name"] == "cell"
    assert preds[0]["score"] == pytest.approx(0.9)
    assert preds[0]["bbox"] == pytest.approx([10, 10, 50, 50])


def test_extraction_still_works_from_a_bare_list(controller):
    assert len(controller.extract_predictions(_results(), "a.png")) == 2


# --- scoring through the trainer ------------------------------------------


def test_an_unannotated_image_scores_real_uncertainty(controller):
    """Before the fix this returned 0.0 with 0 detections on an image the
    model finds two objects in."""
    record = controller.score_image("a.png", "a.png", _Trainer())

    assert record["mode"] == MODE_UNCERTAINTY
    assert record["breakdown"]["detections"] == 2
    assert record["score"] > 0


def test_an_annotated_image_scores_real_disagreement(qtbot):
    win = _Window({"a.png": {"cell": [
        {"segmentation": [10, 10, 60, 10, 60, 60, 10, 60],
         "category_name": "cell", "number": 1}
    ]}})
    qtbot.addWidget(win)
    controller = ReviewController(win)

    record = controller.score_image("a.png", "a.png", _Trainer())

    assert record["mode"] == MODE_DISAGREEMENT
    # The cell prediction matches the label; the nucleus one is spurious.
    assert record["breakdown"]["matched"] == 1
    assert record["breakdown"]["spurious"] == 1
    assert record["breakdown"]["missed"] == 0


def test_perfect_agreement_still_scores_zero(qtbot):
    """Guards the other direction: the unwrap must not inflate scores."""
    win = _Window({"a.png": {
        "cell": [{"bbox": [10, 10, 50, 50], "category_name": "cell", "number": 1}],
        "nucleus": [{"bbox": [100, 100, 60, 60], "category_name": "nucleus", "number": 1}],
    }})
    qtbot.addWidget(win)
    controller = ReviewController(win)

    record = controller.score_image("a.png", "a.png", _Trainer())

    assert record["score"] == pytest.approx(0.0, abs=0.05)
    assert record["breakdown"]["matched"] == 2
