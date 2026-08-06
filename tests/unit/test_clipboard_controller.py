"""Annotation clipboard: copy / paste across images and slices (issue #66).

Driven against a fake main window rather than a real ``ImageAnnotator``: the
clipboard's interesting behaviour is entirely in *what it copies and how it
rewrites it on the way back out* -- deep-copy independence, class resolution,
bounds clamping, pose-schema gating, one-undo-entry-per-paste -- none of which
needs a live canvas.
"""

import pytest
from PyQt6.QtWidgets import QDialog, QWidget

from src.digitalsreeni_image_annotator.controllers.clipboard_controller import (
    CREATE_NEW,
    ClassMappingDialog,
    ClipboardController,
)


class _FakePixmap:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def width(self):
        return self._w

    def height(self):
        return self._h


class _FakeImageLabel:
    def __init__(self, width=200, height=200):
        self.annotations = {}
        self.highlighted_annotations = []
        self.class_colors = {}
        self.original_pixmap = _FakePixmap(width, height)

    def update(self):
        pass


class _FakeAnnotationController:
    def __init__(self):
        self.history_calls = 0
        self.saved = 0
        self.listed = []
        self.selection = None

    def record_history(self, key=None):
        self.history_calls += 1

    def add_annotation_to_list(self, annotation):
        annotation["number"] = len(self.listed) + 1
        self.listed.append(annotation)

    def save_current_annotations(self):
        self.saved += 1

    def apply_canvas_selection(self, annotations, mode):
        self.selection = (list(annotations), mode)


class _FakeWindow(QWidget):
    def __init__(self, classes=("cell",), width=200, height=200, schemas=None):
        super().__init__()
        self.image_label = _FakeImageLabel(width, height)
        self.class_mapping = {name: i + 1 for i, name in enumerate(classes)}
        self.keypoint_schemas = dict(schemas or {})
        self.annotation_controller = _FakeAnnotationController()
        self.added_classes = []

    def add_class(self, name, color=None):
        self.class_mapping[name] = len(self.class_mapping) + 1
        self.added_classes.append(name)

    def update_slice_list_colors(self):
        pass

    def auto_save(self):
        pass


@pytest.fixture
def window(qtbot):
    win = _FakeWindow()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def clipboard(window):
    return ClipboardController(window)


def _square(x0, y0, side, name="cell", number=1):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
        "category_id": 1,
        "number": number,
    }


def _pose(k, name="person", number=1):
    points = [(10 + 5 * i, 10 + 5 * i, 2) for i in range(k)]
    flat = [c for p in points for c in p]
    return {
        "keypoints": flat,
        "num_keypoints": k,
        "bbox": [5, 5, 5 * k + 10, 5 * k + 10],
        "category_name": name,
        "number": number,
    }


# --- copy ------------------------------------------------------------------


def test_copy_with_no_selection_does_not_consume_the_key(clipboard):
    assert clipboard.copy_selection() is False
    assert clipboard.has_content() is False


def test_copy_stores_a_deep_copy(clipboard, window):
    source = _square(10, 10, 20)
    window.image_label.highlighted_annotations = [source]

    assert clipboard.copy_selection() is True

    source["segmentation"][0] = 999
    clipboard.paste()
    pasted = window.image_label.annotations["cell"][0]
    assert pasted["segmentation"][0] == 10, "clipboard aliased the source"


def test_copy_strips_the_per_image_number(clipboard, window):
    window.image_label.highlighted_annotations = [_square(10, 10, 20, number=7)]
    clipboard.copy_selection()
    clipboard.paste()
    # The target image had no annotations, so numbering restarts at 1.
    assert window.image_label.annotations["cell"][0]["number"] == 1


def test_clipboard_survives_switching_image(clipboard, window):
    window.image_label.highlighted_annotations = [_square(10, 10, 20)]
    clipboard.copy_selection()

    # Simulate a navigation: the per-image working copy is replaced wholesale.
    window.image_label.annotations = {}
    window.image_label.highlighted_annotations = []

    assert clipboard.has_content() is True
    assert clipboard.paste() is True
    assert len(window.image_label.annotations["cell"]) == 1


# --- paste geometry --------------------------------------------------------


def test_paste_lands_at_the_original_coordinates(clipboard, window):
    window.image_label.highlighted_annotations = [_square(40, 60, 20)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert window.image_label.annotations["cell"][0]["segmentation"][:2] == [40, 60]


def test_paste_into_a_smaller_image_clamps(qtbot):
    source = _FakeWindow(width=2000, height=2000)
    qtbot.addWidget(source)
    clipboard = ClipboardController(source)
    source.image_label.highlighted_annotations = [_square(1500, 1500, 300)]
    clipboard.copy_selection()

    # Same controller, now pointed at a 100x100 image.
    source.image_label = _FakeImageLabel(100, 100)
    clipboard.paste()

    segmentation = source.image_label.annotations["cell"][0]["segmentation"]
    assert all(0 <= c <= 100 for c in segmentation), segmentation


def test_paste_clamps_the_raw_polygon_too(clipboard, window):
    """If the raw copy kept out-of-bounds coordinates, setting Detail-% back to
    100 would restore them (ADR-025 + ADR-024 interacting)."""
    annotation = _square(10, 10, 20)
    annotation["segmentation_raw"] = [0, 0, 5000, 0, 5000, 5000]
    window.image_label.highlighted_annotations = [annotation]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    raw = window.image_label.annotations["cell"][0]["segmentation_raw"]
    assert all(0 <= c <= 200 for c in raw), raw


def test_pasted_shapes_are_independent_of_each_other(clipboard, window):
    window.image_label.highlighted_annotations = [_square(10, 10, 20)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()
    clipboard.paste()

    first, second = window.image_label.annotations["cell"]
    first["segmentation"][0] = 111
    assert second["segmentation"][0] == 10


# --- undo / selection ------------------------------------------------------


def test_a_multi_annotation_paste_is_one_history_entry(clipboard, window):
    window.image_label.highlighted_annotations = [
        _square(10, 10, 10, number=1),
        _square(30, 30, 10, number=2),
        _square(50, 50, 10, number=3),
        _square(70, 70, 10, number=4),
        _square(90, 90, 10, number=5),
    ]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert window.annotation_controller.history_calls == 1
    assert len(window.image_label.annotations["cell"]) == 5


def test_history_is_recorded_before_any_mutation(clipboard, window):
    """ADR-026: the snapshot has to predate the change or undo restores the
    post-paste state."""
    order = []
    window.annotation_controller.record_history = lambda key=None: order.append("history")
    original_add = window.annotation_controller.add_annotation_to_list

    def spy(annotation):
        order.append("mutate")
        original_add(annotation)

    window.annotation_controller.add_annotation_to_list = spy

    window.image_label.highlighted_annotations = [_square(10, 10, 20)]
    clipboard.copy_selection()
    clipboard.paste()

    assert order == ["history", "mutate"]


def test_pasted_annotations_become_the_selection(clipboard, window):
    window.image_label.highlighted_annotations = [_square(10, 10, 20)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    selection, mode = window.annotation_controller.selection
    assert mode == "replace"
    assert selection == window.image_label.annotations["cell"]


def test_paste_with_an_empty_clipboard_does_not_consume(clipboard):
    assert clipboard.paste() is False


def test_paste_without_an_image_does_not_consume(clipboard, window):
    window.image_label.highlighted_annotations = [_square(10, 10, 20)]
    clipboard.copy_selection()
    window.image_label.original_pixmap = None
    assert clipboard.paste() is False


# --- class resolution ------------------------------------------------------


def test_existing_class_needs_no_dialog(clipboard, window, monkeypatch):
    monkeypatch.setattr(
        ClassMappingDialog, "exec", lambda self: pytest.fail("dialog must not open")
    )
    window.image_label.highlighted_annotations = [_square(10, 10, 20, name="cell")]
    clipboard.copy_selection()
    clipboard.paste()
    assert "cell" in window.image_label.annotations


def test_missing_class_can_be_created(clipboard, window, monkeypatch):
    monkeypatch.setattr(
        ClassMappingDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(ClassMappingDialog, "chosen", lambda self: CREATE_NEW)

    window.image_label.highlighted_annotations = [_square(10, 10, 20, name="mito")]
    clipboard.copy_selection()
    clipboard.paste()

    assert window.added_classes == ["mito"]
    assert "mito" in window.image_label.annotations


def test_missing_class_can_be_mapped_onto_an_existing_one(clipboard, window, monkeypatch):
    monkeypatch.setattr(
        ClassMappingDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(ClassMappingDialog, "chosen", lambda self: "cell")

    window.image_label.highlighted_annotations = [_square(10, 10, 20, name="mito")]
    clipboard.copy_selection()
    clipboard.paste()

    assert window.added_classes == []
    assert "cell" in window.image_label.annotations
    assert "mito" not in window.image_label.annotations
    assert window.image_label.annotations["cell"][0]["category_name"] == "cell"


def test_cancelling_the_class_dialog_pastes_nothing_at_all(clipboard, window, monkeypatch):
    """A partial paste after a cancel is worse than none."""
    monkeypatch.setattr(
        ClassMappingDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    window.image_label.highlighted_annotations = [
        _square(10, 10, 20, name="cell"),
        _square(30, 30, 20, name="mito", number=2),
    ]
    clipboard.copy_selection()
    clipboard.paste()

    assert window.image_label.annotations == {}
    assert window.annotation_controller.history_calls == 0


def test_missing_class_is_asked_about_once_not_once_per_annotation(
    clipboard, window, monkeypatch
):
    calls = []

    def fake_exec(self):
        calls.append(1)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ClassMappingDialog, "exec", fake_exec)
    monkeypatch.setattr(ClassMappingDialog, "chosen", lambda self: CREATE_NEW)

    window.image_label.highlighted_annotations = [
        _square(10 * i, 10, 5, name="mito", number=i) for i in range(1, 6)
    ]
    clipboard.copy_selection()
    clipboard.paste()

    assert len(calls) == 1


# --- pose instances (ADR-029) ----------------------------------------------


def test_pose_pastes_into_a_class_with_a_matching_schema(qtbot):
    schema = {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": [0, 1, 2]}
    window = _FakeWindow(classes=("person",), schemas={"person": schema})
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    window.image_label.highlighted_annotations = [_pose(3)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert len(window.image_label.annotations["person"]) == 1


def test_pose_is_skipped_on_a_k_mismatch(qtbot, monkeypatch):
    from src.digitalsreeni_image_annotator.controllers import clipboard_controller

    warned = []
    monkeypatch.setattr(
        clipboard_controller.QMessageBox, "warning",
        lambda *args, **kwargs: warned.append(args),
    )

    schema = {"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]}
    window = _FakeWindow(classes=("person",), schemas={"person": schema})
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    window.image_label.highlighted_annotations = [_pose(17)]  # K=17 vs schema K=2
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert window.image_label.annotations == {}, "a corrupt instance must not be written"
    assert warned, "the skip must be reported, not silent"


def test_pose_is_skipped_when_the_target_class_has_no_schema(qtbot, monkeypatch):
    from src.digitalsreeni_image_annotator.controllers import clipboard_controller

    monkeypatch.setattr(
        clipboard_controller.QMessageBox, "warning", lambda *a, **k: None
    )
    window = _FakeWindow(classes=("person",))  # no schemas at all
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    window.image_label.highlighted_annotations = [_pose(3)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert window.image_label.annotations == {}


def test_pasted_pose_never_gains_a_segmentation_key(qtbot):
    """Its absence is the discriminator that routes area, Detail-% and
    rendering (ADR-029)."""
    schema = {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": [0, 1, 2]}
    window = _FakeWindow(classes=("person",), schemas={"person": schema})
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    instance = _pose(3)
    instance["segmentation"] = None  # as a sloppy import might leave it
    window.image_label.highlighted_annotations = [instance]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    pasted = window.image_label.annotations["person"][0]
    assert "segmentation" not in pasted


def test_pasted_pose_is_clamped_into_the_target(qtbot):
    schema = {"names": ["a", "b", "c"], "skeleton": [], "flip_idx": [0, 1, 2]}
    window = _FakeWindow(classes=("person",), schemas={"person": schema}, width=20, height=20)
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    window.image_label.highlighted_annotations = [_pose(3)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    keypoints = window.image_label.annotations["person"][0]["keypoints"]
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    assert all(0 <= v <= 20 for v in xs + ys), keypoints


def test_a_mixed_paste_keeps_the_valid_shapes(qtbot, monkeypatch):
    """One bad pose must not cost the user the polygons pasted alongside it."""
    from src.digitalsreeni_image_annotator.controllers import clipboard_controller

    monkeypatch.setattr(
        clipboard_controller.QMessageBox, "warning", lambda *a, **k: None
    )
    window = _FakeWindow(classes=("cell", "person"))
    qtbot.addWidget(window)
    clipboard = ClipboardController(window)

    window.image_label.highlighted_annotations = [_square(10, 10, 20), _pose(3)]
    clipboard.copy_selection()
    window.image_label.annotations = {}
    clipboard.paste()

    assert len(window.image_label.annotations["cell"]) == 1
    assert "person" not in window.image_label.annotations
