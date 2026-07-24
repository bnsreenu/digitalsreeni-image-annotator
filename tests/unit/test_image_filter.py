"""
Unit tests for the image-list annotation-status filter (upstream issue #27).

Covers ImageController.image_has_annotations and apply_image_filter.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QListWidget, QComboBox

from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)


FILTER_ALL = 0
FILTER_WITHOUT = 1
FILTER_WITH = 2


class FakeMainWindow(QWidget):
    """Minimal stand-in for ImageAnnotator with the state the filter reads."""


@pytest.fixture
def mw(qtbot):
    window = FakeMainWindow()
    qtbot.addWidget(window)
    window.image_list = QListWidget(window)
    window.image_filter_combo = QComboBox(window)
    window.image_filter_combo.addItems(
        ["All images", "Without annotations", "With annotations"]
    )
    window.image_group_combo = QComboBox(window)
    window.image_group_combo.addItem("All groups")
    window.all_images = []
    window.all_annotations = {}
    window.image_slices = {}
    window.image_controller = ImageController(window)
    return window


def _add_image(mw, file_name, is_multi_slice=False):
    mw.all_images.append(
        {"file_name": file_name, "is_multi_slice": is_multi_slice}
    )
    mw.image_list.addItem(file_name)


class TestImageHasAnnotations:
    def test_regular_image_without_annotations(self, mw):
        _add_image(mw, "plain.png")
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_regular_image_with_empty_class_lists(self, mw):
        _add_image(mw, "plain.png")
        mw.all_annotations["plain.png"] = {"cell": []}
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_regular_image_with_annotations(self, mw):
        _add_image(mw, "plain.png")
        mw.all_annotations["plain.png"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_with_annotated_slice(self, mw):
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.image_slices["stack"] = [("stack_T1_Z1", None), ("stack_T1_Z2", None)]
        mw.all_annotations["stack_T1_Z2"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_without_annotated_slices(self, mw):
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.image_slices["stack"] = [("stack_T1_Z1", None)]
        mw.all_annotations["stack_T1_Z1"] = {"cell": []}
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_prefix_fallback_when_slices_not_loaded(self, mw):
        # Project annotations exist under slice keys, but the slices were
        # never extracted (e.g. load cancelled) — prefix fallback applies.
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.all_annotations["stack_T1_Z5_C1"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_no_substring_false_positive(self, mw):
        # "bee" must not match keys of "honeybee" (and vice versa).
        _add_image(mw, "bee.tif", is_multi_slice=True)
        mw.all_annotations["honeybee_T1_Z1"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])


class TestApplyImageFilter:
    @pytest.fixture
    def populated(self, mw):
        _add_image(mw, "annotated.png")
        _add_image(mw, "empty.png")
        mw.all_annotations["annotated.png"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        # Select the annotated image so the "never hide current" rule is
        # exercised by a dedicated test, not by accident here.
        mw.image_list.setCurrentRow(-1)
        return mw

    def _hidden(self, mw):
        return [
            mw.image_list.isRowHidden(i) for i in range(mw.image_list.count())
        ]

    def test_all_images_shows_everything(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_ALL)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, False]

    def test_without_annotations_hides_annotated(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITHOUT)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [True, False]

    def test_with_annotations_hides_unannotated(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITH)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, True]

    def test_current_row_is_hidden_when_not_matching(self, populated):
        # The current row is not exempt: a non-matching selected image
        # leaves the list (the canvas keeps showing it — see the wiring
        # test for the switch_image / canvas-unchanged guarantee).
        populated.image_list.setCurrentRow(0)  # annotated.png
        populated.image_filter_combo.setCurrentIndex(FILTER_WITHOUT)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [True, False]

    def test_no_combo_is_a_noop(self, mw):
        _add_image(mw, "plain.png")
        del mw.image_filter_combo
        mw.image_controller.apply_image_filter()  # must not raise
        assert not mw.image_list.isRowHidden(0)

    def test_switching_back_to_all_unhides(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITH)
        populated.image_controller.apply_image_filter()
        populated.image_filter_combo.setCurrentIndex(FILTER_ALL)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, False]


class TestGroupFilter:
    """Group filter (issue #43): a specific group hides non-members;
    combines with the status filter via OR; index 0 of both hides nothing.
    """

    def _visible(self, mw):
        return [
            mw.image_list.item(i).text()
            for i in range(mw.image_list.count())
            if not mw.image_list.isRowHidden(i)
        ]

    def _add(self, mw, name, group=None, annotated=False):
        info = {"file_name": name, "is_multi_slice": False}
        if group:
            info["group"] = group
        mw.all_images.append(info)
        if annotated:
            mw.all_annotations[name] = {"cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]}

    def test_group_filter_hides_non_members(self, mw):
        self._add(mw, "a.png", group="G1")
        self._add(mw, "b.png", group="G2")
        self._add(mw, "c.png")  # ungrouped
        mw.image_controller.sort_image_list()  # rebuilds list + group combo
        mw.image_list.setCurrentRow(-1)

        idx = mw.image_group_combo.findText("G1")
        assert idx > 0  # combo was populated with the derived groups
        mw.image_group_combo.setCurrentIndex(idx)
        mw.image_controller.apply_image_filter()

        assert self._visible(mw) == ["a.png"]

    def test_combined_status_and_group(self, mw):
        self._add(mw, "g1_annot.png", group="G1", annotated=True)
        self._add(mw, "g1_plain.png", group="G1")
        self._add(mw, "g2_annot.png", group="G2", annotated=True)
        mw.image_controller.sort_image_list()
        mw.image_list.setCurrentRow(-1)

        mw.image_filter_combo.setCurrentIndex(FILTER_WITH)
        mw.image_group_combo.setCurrentIndex(mw.image_group_combo.findText("G1"))
        mw.image_controller.apply_image_filter()

        # "With annotations" drops g1_plain; group G1 drops g2_annot.
        assert self._visible(mw) == ["g1_annot.png"]

    def test_both_combos_index_zero_hides_nothing(self, mw):
        self._add(mw, "a.png", group="G1")
        self._add(mw, "b.png")
        mw.image_controller.sort_image_list()
        mw.image_list.setCurrentRow(-1)

        mw.image_filter_combo.setCurrentIndex(FILTER_ALL)
        mw.image_group_combo.setCurrentIndex(0)
        mw.image_controller.apply_image_filter()

        assert self._visible(mw) == ["b.png", "a.png"]  # ungrouped clusters first
        assert not any(
            mw.image_list.isRowHidden(i) for i in range(mw.image_list.count())
        )
