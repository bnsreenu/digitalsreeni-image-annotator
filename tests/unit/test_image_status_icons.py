"""
Unit tests for image-list annotation-status badges (issue #43).

ImageController.refresh_image_status_icons paints a filled dot on
annotated rows and a hollow dot on un-annotated ones, cached per
(annotated, dark_mode) and rebuilt on a theme flip.
"""

from PyQt6.QtWidgets import QListWidget, QWidget

from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)


class FakeMainWindow(QWidget):
    pass


def _make_window():
    window = FakeMainWindow()
    window.image_list = QListWidget(window)
    window.all_images = []
    window.all_annotations = {}
    window.image_slices = {}
    window.dark_mode = False
    window.image_controller = ImageController(window)
    return window


def _add_image(window, name, annotated=False):
    window.all_images.append({"file_name": name, "is_multi_slice": False})
    window.image_list.addItem(name)
    if annotated:
        window.all_annotations[name] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }


def test_annotated_and_unannotated_rows_get_distinct_icons(qt_application):
    window = _make_window()
    _add_image(window, "annotated.png", annotated=True)
    _add_image(window, "empty.png")

    window.image_controller.refresh_image_status_icons()

    annotated_icon = window.image_list.item(0).icon()
    empty_icon = window.image_list.item(1).icon()
    assert not annotated_icon.isNull()
    assert not empty_icon.isNull()

    annotated_img = annotated_icon.pixmap(12, 12).toImage()
    empty_img = empty_icon.pixmap(12, 12).toImage()
    assert annotated_img != empty_img

    window.deleteLater()


def test_icons_are_cached_per_state(qt_application):
    window = _make_window()
    _add_image(window, "a.png", annotated=True)

    window.image_controller.refresh_image_status_icons()
    cache = window.image_controller._status_icon_cache
    assert (True, False) in cache  # annotated, light theme

    cached_icon = cache[(True, False)]
    window.image_controller.refresh_image_status_icons()
    # Second refresh reuses the same cached QIcon object.
    assert cache[(True, False)] is cached_icon

    window.deleteLater()


def test_theme_flip_repaints_icons_with_theme_tuned_colors(qt_application):
    # The badge colours are theme-tuned, so a dark-mode flip must produce a
    # visibly DIFFERENT pixmap (not just a rebuilt cache key). This defends
    # against the dark-mode machinery silently becoming inert.
    window = _make_window()
    _add_image(window, "a.png")  # un-annotated

    window.image_controller.refresh_image_status_icons()
    light_img = window.image_list.item(0).icon().pixmap(12, 12).toImage()
    assert (False, False) in window.image_controller._status_icon_cache

    window.dark_mode = True
    window.image_controller.on_theme_changed()

    cache = window.image_controller._status_icon_cache
    assert (False, True) in cache  # rebuilt for the dark theme
    assert (False, False) not in cache  # stale light-theme entry cleared

    dark_img = window.image_list.item(0).icon().pixmap(12, 12).toImage()
    assert dark_img != light_img  # the dot actually recolours per theme

    window.deleteLater()
