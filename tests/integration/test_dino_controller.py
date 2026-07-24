"""DINOController workflow tests (issue #37).

The DINO -> SAM text-prompted annotation pipeline encodes several subtle
invariants that have already caused real bugs (masks bleeding across slices,
slices silently skipped in batch, auto-accept forgotten in the single path).
These build one real ImageAnnotator (offscreen) and drive
``window.dino_controller`` directly.

Only the inference boundary (``dino_utils.detect`` /
``sam_utils.apply_sam_predictions_batch``) and the modal ``QMessageBox``
statics are mocked -- no model weights, no network, no worker threads.
Annotation dicts, ``dino_batch_results``, ``temp_annotations`` and the class
mapping are all real state.
"""

import pytest


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


@pytest.fixture
def dino_ready(window, monkeypatch):
    """Window primed for DINO detection with fully mocked inference."""
    from PyQt6.QtGui import QImage, QColor
    from PyQt6.QtWidgets import QMessageBox

    import digitalsreeni_image_annotator.core.torch_utils as torch_utils

    c = window.dino_controller
    window.dino_model_loaded = True
    window.sam_utils.current_sam_model = "SAM 2 tiny"       # gate only, never used
    monkeypatch.setattr(c, "_ensure_dino_model_downloaded", lambda name: True)
    monkeypatch.setattr(
        c, "_build_dino_class_configs",
        lambda: [{"name": "cell", "phrases": ["cell"],
                  "box_thr": 0.3, "txt_thr": 0.25, "nms_thr": 0.5}],
    )
    # maybe_warn_cpu_fallback is imported *inside* the method bodies, so patch
    # it at the source module, not the controller module.
    monkeypatch.setattr(torch_utils, "maybe_warn_cpu_fallback", lambda *a, **k: None)

    # Modal dialogs hang offscreen -- patch every static that a path can reach.
    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))
    # auto_save() pops a question if no project file -- answer No so it no-ops.
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    # A current image + one known class.
    img = QImage(64, 64, QImage.Format.Format_RGB888)
    img.fill(0)
    window.current_image = img
    window.image_file_name = "img1.png"
    window.current_slice = None
    window.class_mapping = {"cell": 1}
    window.image_label.class_colors["cell"] = QColor(255, 0, 0)

    # Canned inference results (one "cell" detection -> one segmentation).
    monkeypatch.setattr(
        window.dino_utils, "detect",
        lambda *a, **k: [{"class_name": "cell", "score": 0.9, "bbox": [1, 1, 10, 10]}],
    )
    monkeypatch.setattr(
        window.sam_utils, "apply_sam_predictions_batch",
        lambda img, boxes: [{"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0]}
                            for _ in boxes],
    )
    return window


# --- helpers ---------------------------------------------------------------

def _temp_ann(category="cell"):
    return {
        "segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0],
        "category_name": category,
        "score": 0.9,
        "source": "dino",
        "temp": True,
    }


def _seed_batch(window, tmp_path):
    """Seed one regular image (real PNG on disk) + one multi-slice image with
    a single materialised slice. Returns the work-item names."""
    from PyQt6.QtGui import QImage
    from PIL import Image

    reg = tmp_path / "reg1.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(reg)

    qimg = QImage(8, 8, QImage.Format.Format_RGB888)
    qimg.fill(0)

    window.all_images = [
        {"file_name": "reg1.png"},
        {"file_name": "stack.tif", "is_multi_slice": True},
    ]
    window.image_paths = {"reg1.png": str(reg)}
    window.image_slices = {"stack": [("stack_T0_Z0", qimg)]}
    return ["reg1.png", "stack_T0_Z0"]


# --- single detection ------------------------------------------------------

def test_single_review_attaches_temp_annotations(dino_ready):
    window = dino_ready
    c = window.dino_controller
    # dino_batch_mode left at the default "Review before accepting".

    c.run_dino_detection_single()

    temp = window.image_label.temp_annotations
    assert len(temp) == 1
    assert temp[0]["source"] == "dino"
    assert temp[0]["temp"] is True
    assert temp[0]["category_name"] == "cell"
    # Review mode must not commit anything to the canvas.
    assert "cell" not in window.image_label.annotations


def test_single_auto_accept_commits_and_numbers(dino_ready):
    window = dino_ready
    c = window.dino_controller
    window.dino_batch_mode.setCurrentText("Auto-accept all detections")

    c.run_dino_detection_single()
    assert window.image_label.temp_annotations == []
    cells = window.image_label.annotations.get("cell", [])
    assert len(cells) == 1
    assert cells[0]["number"] == 1

    # A second run appends and continues the per-class numbering.
    c.run_dino_detection_single()
    cells = window.image_label.annotations.get("cell", [])
    assert len(cells) == 2
    assert cells[1]["number"] == 2


def test_single_sam_error_filtered_review_mode(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    monkeypatch.setattr(
        window.sam_utils, "apply_sam_predictions_batch",
        lambda img, boxes: [{"error": "boom"} for _ in boxes],
    )

    c.run_dino_detection_single()  # review mode

    assert window.image_label.temp_annotations == []


def test_single_sam_error_filtered_auto_accept(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    window.dino_batch_mode.setCurrentText("Auto-accept all detections")
    monkeypatch.setattr(
        window.sam_utils, "apply_sam_predictions_batch",
        lambda img, boxes: [{"error": "boom"} for _ in boxes],
    )

    c.run_dino_detection_single()

    assert "cell" not in window.image_label.annotations


def test_single_empty_results_no_temp(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    monkeypatch.setattr(window.dino_utils, "detect", lambda *a, **k: [])

    c.run_dino_detection_single()

    assert window.image_label.temp_annotations == []
    assert window.btn_detect_single.isEnabled()


def test_single_none_results_no_temp(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    monkeypatch.setattr(window.dino_utils, "detect", lambda *a, **k: None)

    c.run_dino_detection_single()

    assert window.image_label.temp_annotations == []
    assert window.btn_detect_single.isEnabled()


# --- guard clauses ---------------------------------------------------------

def test_guard_no_dino_model_skips_detect(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    window.dino_model_loaded = False
    calls = []
    monkeypatch.setattr(window.dino_utils, "detect", lambda *a, **k: calls.append(1))

    c.run_dino_detection_single()

    assert calls == []


def test_guard_no_sam_model_skips_detect(dino_ready, monkeypatch):
    window = dino_ready
    c = window.dino_controller
    window.sam_utils.current_sam_model = None
    calls = []
    monkeypatch.setattr(window.dino_utils, "detect", lambda *a, **k: calls.append(1))

    c.run_dino_detection_single()

    assert calls == []


# --- batch work-item flattening -------------------------------------------

def test_collect_batch_work_items_flattens_slices_and_skips(window, tmp_path):
    from PyQt6.QtGui import QImage
    from PIL import Image

    c = window.dino_controller

    reg_ok = tmp_path / "reg1.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(reg_ok)

    qimg = QImage(8, 8, QImage.Format.Format_RGB888)
    qimg.fill(0)

    window.all_images = [
        {"file_name": "reg1.png"},                          # regular, present
        {"file_name": "stack.tif", "is_multi_slice": True},  # 2 slices
        {"file_name": "missing.png"},                       # path missing -> skip
        {"file_name": "empty.tif", "is_multi_slice": True},  # no slices -> skip
    ]
    window.image_paths = {"reg1.png": str(reg_ok)}          # missing.png absent
    window.image_slices = {
        "stack": [("stack_T0_Z0", qimg), ("stack_T0_Z1", qimg)],
        # "empty" absent -> no materialised slices
    }

    items = c._collect_dino_batch_work_items()

    names = [n for n, _ in items]
    assert names == ["reg1.png", "stack_T0_Z0", "stack_T0_Z1"]
    assert all(isinstance(img, QImage) for _, img in items)


# --- batch detection -------------------------------------------------------

def test_batch_review_stores_results(dino_ready, monkeypatch, tmp_path):
    window = dino_ready
    c = window.dino_controller
    names = _seed_batch(window, tmp_path)
    window.dino_batch_mode.setCurrentText("Review before accepting")

    recorded = []
    monkeypatch.setattr(c, "_show_dino_batch_review", lambda: recorded.append(True))

    c.run_dino_detection_batch()

    assert set(window.dino_batch_results) == set(names)
    for results in window.dino_batch_results.values():
        assert results
        assert results[0]["source"] == "dino"
        assert results[0]["temp"] is True
        assert "segmentation" in results[0]
    # Nothing committed to the canvas in review mode.
    assert window.image_label.annotations == {}
    assert recorded == [True]


def test_batch_auto_accept_commits_including_slices(dino_ready, tmp_path):
    window = dino_ready
    c = window.dino_controller
    _seed_batch(window, tmp_path)
    window.dino_batch_mode.setCurrentText("Auto-accept all detections")

    c.run_dino_detection_batch()

    # Neither work item is the on-screen image ("img1.png"), so both land in
    # all_annotations via _commit_dino_results' non-current branch.
    assert "cell" in window.all_annotations.get("reg1.png", {})
    assert window.all_annotations["reg1.png"]["cell"][0]["number"] == 1
    assert "cell" in window.all_annotations.get("stack_T0_Z0", {})


# --- temp re-sync on image/slice switch (the bleed-bug regression) ---------

def test_refresh_dino_temp_clears_stale_masks_on_switch(window):
    c = window.dino_controller
    stored = [_temp_ann()]
    window.dino_batch_results = {"img2.png": stored}
    window.image_file_name = "img2.png"
    window.current_slice = None

    c._refresh_dino_temp_for_current()

    # Synced to a *copy* of the stored list, not the same object.
    assert window.image_label.temp_annotations == stored
    assert window.image_label.temp_annotations is not stored

    # Switching to an image with no pending entry clears the field, so the
    # previous image's masks don't bleed onto every slice.
    window.image_file_name = "other.png"
    c._refresh_dino_temp_for_current()
    assert window.image_label.temp_annotations == []


# --- accept / reject -------------------------------------------------------

def test_accept_dino_results_commits_and_pops(dino_ready):
    window = dino_ready
    c = window.dino_controller
    temp = _temp_ann("cell")
    window.image_label.temp_annotations = [temp]
    window.dino_batch_results = {"img1.png": [temp]}
    window.image_file_name = "img1.png"
    window.current_slice = None

    c.accept_dino_results()

    cells = window.image_label.annotations.get("cell", [])
    assert len(cells) == 1
    assert cells[0]["segmentation"] == temp["segmentation"]
    assert window.image_label.temp_annotations == []
    assert "img1.png" not in window.dino_batch_results


def test_accept_dino_results_skips_unknown_class(dino_ready):
    window = dino_ready
    c = window.dino_controller
    window.image_label.temp_annotations = [_temp_ann("ghost")]  # not in class_mapping
    window.image_file_name = "img1.png"
    window.current_slice = None

    c.accept_dino_results()  # must not raise

    assert "ghost" not in window.image_label.annotations
    assert window.image_label.temp_annotations == []


def test_reject_dino_results_clears_without_commit(dino_ready):
    window = dino_ready
    c = window.dino_controller
    temp = _temp_ann("cell")
    window.image_label.temp_annotations = [temp]
    window.dino_batch_results = {"img1.png": [temp]}
    window.image_file_name = "img1.png"
    window.current_slice = None

    c.reject_dino_results()

    assert window.image_label.temp_annotations == []
    assert "img1.png" not in window.dino_batch_results
    assert "cell" not in window.image_label.annotations


# --- Temp-* class review flow (shared with YOLO predictions) ---------------

def test_temp_class_accept_moves_to_permanent(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    for m in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    c = window.dino_controller
    ann = {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0],
           "category_name": "Temp-foo", "number": 1}

    c.add_temp_classes({"Temp-foo": [ann]})
    assert "Temp-foo" in window.image_label.class_colors
    assert "Temp-foo" in window.image_label.annotations

    c.accept_visible_temp_classes()

    assert "foo" in window.image_label.annotations
    assert len(window.image_label.annotations["foo"]) == 1
    moved = window.image_label.annotations["foo"][0]
    assert moved["category_name"] == "foo"
    assert moved["number"] == 1
    assert "Temp-foo" not in window.image_label.annotations
    assert "Temp-foo" not in window.image_label.class_colors


def test_temp_class_reject_deletes(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    for m in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, m, staticmethod(lambda *a, **k: None))

    c = window.dino_controller
    ann = {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0],
           "category_name": "Temp-foo", "number": 1}

    c.add_temp_classes({"Temp-foo": [ann]})
    c.reject_visible_temp_classes()

    assert "Temp-foo" not in window.image_label.annotations
    assert "Temp-foo" not in window.image_label.class_colors


def test_check_temp_annotations_yes_discards(window, monkeypatch):
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )

    c = window.dino_controller
    window.image_label.annotations["Temp-foo"] = [
        {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0], "category_name": "Temp-foo"}
    ]
    window.image_label.class_colors["Temp-foo"] = QColor(1, 2, 3)

    result = c.check_temp_annotations()

    assert result is True
    assert "Temp-foo" not in window.image_label.annotations


def test_check_temp_annotations_no_keeps(window, monkeypatch):
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )

    c = window.dino_controller
    window.image_label.annotations["Temp-foo"] = [
        {"segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 10.0], "category_name": "Temp-foo"}
    ]
    window.image_label.class_colors["Temp-foo"] = QColor(1, 2, 3)

    result = c.check_temp_annotations()

    assert result is False
    assert "Temp-foo" in window.image_label.annotations


# --- DINOReviewEventFilter gating -----------------------------------------

def test_event_filter_accepts_pending_dino_on_enter(window, monkeypatch):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    called = []
    monkeypatch.setattr(window, "accept_dino_results", lambda: called.append("accept"))
    window.image_label.temp_annotations = [{"source": "dino", "category_name": "cell"}]

    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                   Qt.KeyboardModifier.NoModifier)
    handled = window._dino_review_filter.eventFilter(window.image_label, ev)

    assert handled is True
    assert called == ["accept"]


def test_event_filter_rejects_pending_dino_on_escape(window, monkeypatch):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    called = []
    monkeypatch.setattr(window, "reject_dino_results", lambda: called.append("reject"))
    window.image_label.temp_annotations = [{"source": "dino", "category_name": "cell"}]

    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                   Qt.KeyboardModifier.NoModifier)
    handled = window._dino_review_filter.eventFilter(window.image_label, ev)

    assert handled is True
    assert called == ["reject"]


def test_event_filter_ignores_when_no_dino_pending(window):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    # Empty temp_annotations -> not consumed.
    window.image_label.temp_annotations = []
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                   Qt.KeyboardModifier.NoModifier)
    assert window._dino_review_filter.eventFilter(window.image_label, ev) is False

    # Pending temp but no dino source (e.g. a SAM overlay) -> not consumed.
    window.image_label.temp_annotations = [{"source": "sam"}]
    ev2 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                    Qt.KeyboardModifier.NoModifier)
    assert window._dino_review_filter.eventFilter(window.image_label, ev2) is False
