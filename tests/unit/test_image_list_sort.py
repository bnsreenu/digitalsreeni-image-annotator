"""
Unit tests for alphabetical image-list sorting (upstream issue #60).

sort_image_list must order the list case-insensitively, keep the
all_images model aligned with the list rows (positional invariant used
by COCO import), and never fire a spurious switch_image.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QListWidget, QComboBox

from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)


class FakeMainWindow(QWidget):
    pass


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
    window.image_paths = {}
    window.is_loading_project = False
    window.auto_save = lambda: None
    window.image_controller = ImageController(window)
    return window


def _populate(mw, names):
    # Populate out of order, mimicking the model+view pairing that
    # add_images_to_list produces before a sort.
    for n in names:
        mw.all_images.append({"file_name": n, "is_multi_slice": False})
        mw.image_list.addItem(n)


def _list_texts(mw):
    return [mw.image_list.item(i).text() for i in range(mw.image_list.count())]


def test_sorts_alphabetically(mw):
    _populate(mw, ["banana.png", "apple.png", "cherry.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == ["apple.png", "banana.png", "cherry.png"]


def test_sort_is_case_insensitive(mw):
    _populate(mw, ["Zebra.png", "apple.png", "Banana.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == ["apple.png", "Banana.png", "Zebra.png"]


def test_model_and_view_stay_aligned(mw):
    _populate(mw, ["d.png", "a.png", "c.png", "b.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == [info["file_name"] for info in mw.all_images]


def test_sort_fires_no_switch_image(mw):
    _populate(mw, ["b.png", "a.png"])
    calls = []
    mw.switch_image = lambda item: calls.append(item)
    mw.image_controller.switch_image = lambda item: calls.append(item)
    mw.image_list.setCurrentRow(0)
    calls.clear()
    mw.image_controller.sort_image_list()  # no select_name, do_switch=False
    assert calls == []


def test_selection_preserved_across_sort(mw):
    _populate(mw, ["b.png", "a.png", "c.png"])
    mw.image_list.setCurrentRow(0)  # b.png
    mw.image_controller.sort_image_list()
    assert mw.image_list.currentItem().text() == "b.png"


def test_select_name_and_switch(mw):
    _populate(mw, ["b.png", "a.png"])
    calls = []
    mw.switch_image = lambda item: calls.append(item.text())
    mw.image_controller.switch_image = lambda item: calls.append(item.text())
    mw.image_controller.sort_image_list(select_name="a.png", do_switch=True)
    assert mw.image_list.currentItem().text() == "a.png"
    assert calls == ["a.png"]


def _populate_grouped(mw, entries):
    # entries: list of (file_name, group_or_None).
    for name, group in entries:
        info = {"file_name": name, "is_multi_slice": False}
        if group:
            info["group"] = group
        mw.all_images.append(info)
        mw.image_list.addItem(name)


def test_grouped_images_cluster_then_sort_by_name(mw):
    _populate_grouped(
        mw,
        [
            ("z.png", None),
            ("b.png", "Beta"),
            ("a.png", "Alpha"),
            ("y.png", None),
            ("c.png", "Alpha"),
        ],
    )
    mw.image_controller.sort_image_list()
    # Ungrouped first (by name), then Alpha, then Beta.
    assert _list_texts(mw) == ["y.png", "z.png", "a.png", "c.png", "b.png"]


def test_grouped_sort_keeps_positional_invariant(mw):
    _populate_grouped(
        mw, [("z.png", None), ("a.png", "Alpha"), ("b.png", "Beta")]
    )
    mw.image_controller.sort_image_list()
    # all_images[i] <-> image_list.item(i) still aligned, and item text
    # stays the bare file name (no group decoration).
    assert _list_texts(mw) == [info["file_name"] for info in mw.all_images]
    assert _list_texts(mw) == ["z.png", "a.png", "b.png"]


def test_group_combo_repopulated_from_derived_groups(mw):
    _populate_grouped(
        mw, [("a.png", "Alpha"), ("b.png", "Beta"), ("c.png", None)]
    )
    mw.image_controller.sort_image_list()
    combo = mw.image_group_combo
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts == ["All groups", "Alpha", "Beta"]


def test_set_image_group_fires_no_switch_image(mw):
    _populate_grouped(mw, [("a.png", None), ("b.png", None)])
    switch_calls = []
    mw.switch_image = lambda item: switch_calls.append(item)
    mw.image_controller.switch_image = lambda item: switch_calls.append(item)
    save_calls = []
    mw.auto_save = lambda: save_calls.append(1)
    mw.image_list.setCurrentRow(0)
    switch_calls.clear()

    mw.image_controller.set_image_group("a.png", "G1")

    assert switch_calls == []  # re-sort must not fire switch_image
    info = next(i for i in mw.all_images if i["file_name"] == "a.png")
    assert info["group"] == "G1"
    assert save_calls  # auto_save fired (not loading a project)


def test_set_image_group_strips_and_clears(mw):
    _populate_grouped(mw, [("a.png", None)])
    info = mw.all_images[0]

    mw.image_controller.set_image_group("a.png", "  Team A  ")
    assert info["group"] == "Team A"  # whitespace stripped

    mw.image_controller.set_image_group("a.png", "   ")
    assert "group" not in info  # blank removes the key

    mw.image_controller.set_image_group("a.png", "Team A")
    mw.image_controller.set_image_group("a.png", None)
    assert "group" not in info  # None removes the key


def test_set_image_group_skips_autosave_during_load(mw):
    _populate_grouped(mw, [("a.png", None)])
    mw.is_loading_project = True
    save_calls = []
    mw.auto_save = lambda: save_calls.append(1)

    mw.image_controller.set_image_group("a.png", "G1")

    assert save_calls == []  # guarded during project load
    assert mw.all_images[0]["group"] == "G1"  # assignment still happens


def test_project_load_path_ends_sorted(mw):
    # Contract: during project load add_images_to_list does NOT sort per
    # image (avoids O(n^2)); the list is rebuilt once afterwards via the
    # update_ui -> update_image_list call. This guards a refactor of that
    # call from silently leaving the post-load list unsorted.
    mw.is_loading_project = True
    mw.image_controller.add_images_to_list(["c.png", "a.png", "b.png"])
    # Not populated/sorted yet while loading.
    assert mw.image_list.count() == 0

    mw.is_loading_project = False
    mw.image_controller.update_image_list()  # what update_ui triggers

    texts = [mw.image_list.item(i).text() for i in range(mw.image_list.count())]
    assert texts == ["a.png", "b.png", "c.png"]
    assert texts == [info["file_name"] for info in mw.all_images]
