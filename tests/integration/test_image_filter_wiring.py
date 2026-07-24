"""
Integration test for the image-filter re-apply wiring (upstream #27).

The unit tests call apply_image_filter() directly; this test goes through
the real mutation path instead: ImageLabel.annotationsBatchSaved →
ImageAnnotator._on_annotations_batch_saved → save_current_annotations →
ClassController.update_slice_list_colors → apply_image_filter. It is the
test that fails if someone refactors the slice-color path and silently
detaches the filter from annotation mutations.

Constructs one full ImageAnnotator (offscreen) — deliberately, despite
the runtime cost, because the coupling under test spans window, signals
and three controllers.
"""

import pytest
from PyQt6.QtCore import Qt


FILTER_WITHOUT = 1  # combo index: "Without annotations"
FILTER_WITH = 2  # combo index: "With annotations"


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def test_annotation_commit_path_reapplies_filter(window):
    # Two regular images, neither annotated yet. currentRow stays -1 so
    # the "never hide current row" exemption doesn't mask the assertion.
    for name in ("a.png", "b.png"):
        window.all_images.append({"file_name": name, "is_multi_slice": False})
        window.image_list.addItem(name)

    window.image_filter_combo.setCurrentIndex(FILTER_WITH)
    assert window.image_list.isRowHidden(0)
    assert window.image_list.isRowHidden(1)

    # Simulate finishing an annotation on a.png the way the canvas does:
    # ImageLabel holds the in-progress annotations and emits the batch
    # finalizer signal. No direct apply_image_filter / all_annotations
    # manipulation here — that's the point of the test.
    window.image_file_name = "a.png"
    window.image_label.annotations = {
        "cell": [{"segmentation": [0, 0, 10, 0, 10, 10], "category_name": "cell"}]
    }
    window.image_label.annotationsBatchSaved.emit()

    assert window.all_annotations["a.png"]  # save path ran
    assert not window.image_list.isRowHidden(0)  # a.png now annotated
    assert window.image_list.isRowHidden(1)  # b.png still hidden


def test_hiding_current_row_keeps_canvas_and_fires_no_switch(window):
    # Hiding the currently selected (non-matching) row must not change
    # the displayed image or fire switch_image — the canvas stays on the
    # worked-on image while its row leaves the list.
    for name in ("annot.png", "plain.png"):
        window.all_images.append({"file_name": name, "is_multi_slice": False})
        window.image_list.addItem(name)
    window.all_annotations["annot.png"] = {
        "cell": [{"segmentation": [0, 0, 1, 0, 1, 1], "category_name": "cell"}]
    }

    # Pure counter (does NOT delegate to the real switch_image, which
    # needs a loaded project): isolates whether the *filter* fires a
    # switch. setCurrentRow below legitimately fires it once via
    # currentRowChanged — that is product behavior and is cleared away.
    calls = []
    window.switch_image = lambda item: calls.append(item)

    window.image_list.setCurrentRow(0)  # select the annotated image
    sentinel = object()
    window.current_image = sentinel
    calls.clear()

    # "Without annotations" must hide row 0 even though it is current.
    window.image_filter_combo.setCurrentIndex(FILTER_WITHOUT)

    assert window.image_list.isRowHidden(0)  # current row hidden
    assert not window.image_list.isRowHidden(1)
    assert window.current_image is sentinel  # canvas unchanged
    assert calls == []  # hiding the current row fired no switch_image


def test_annotation_commit_sets_status_badge(window):
    # The same real mutation path (issue #43): finishing an annotation must
    # refresh the per-row status badge via update_slice_list_colors ->
    # apply_image_filter -> refresh_image_status_icons.
    for name in ("a.png", "b.png"):
        window.all_images.append({"file_name": name, "is_multi_slice": False})
        window.image_list.addItem(name)
    window.image_controller.refresh_image_status_icons()

    def _icon_for(name):
        items = window.image_list.findItems(name, Qt.MatchFlag.MatchExactly)
        return items[0].icon()

    window.image_file_name = "a.png"
    window.image_label.annotations = {
        "cell": [{"segmentation": [0, 0, 10, 0, 10, 10], "category_name": "cell"}]
    }
    window.image_label.annotationsBatchSaved.emit()

    icon_a = _icon_for("a.png")
    icon_b = _icon_for("b.png")
    assert not icon_a.isNull()  # badge was set on the annotated row
    # Annotated (green) vs un-annotated (hollow) badges differ.
    assert icon_a.pixmap(12, 12).toImage() != icon_b.pixmap(12, 12).toImage()


def test_group_survives_save_reload_roundtrip(window, tmp_path):
    from PyQt6.QtGui import QImage

    # A real (tiny) image on disk so the load path can resolve it.
    img_path = tmp_path / "photo.png"
    QImage(4, 4, QImage.Format.Format_RGB32).save(str(img_path))

    # Stub auto_save so add/assign don't pop a blocking Save dialog.
    window.auto_save = lambda *a, **k: None

    window.image_controller.add_images_to_list([str(img_path)])
    window.image_controller.set_image_group("photo.png", "Batch A")
    info = next(i for i in window.all_images if i["file_name"] == "photo.png")
    assert info["group"] == "Batch A"

    project_data = window.project_controller.build_project_data()
    assert any(
        img.get("file_name") == "photo.png" and img.get("group") == "Batch A"
        for img in project_data["images"]
    )

    # Reload the saved data into the same window (open-project data contract).
    window.is_loading_project = True
    try:
        window.project_controller.load_project_data(project_data)
    finally:
        window.is_loading_project = False

    reloaded = next(i for i in window.all_images if i["file_name"] == "photo.png")
    assert reloaded.get("group") == "Batch A"
