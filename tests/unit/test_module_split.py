"""Lock the compatibility contract of the image_label.py split (issue #46).

The seven pure geometry helpers moved to widgets/edit_gestures.py as module-level
functions; ImageLabel keeps class-level ``staticmethod`` aliases so the historical
``ImageLabel._resize_bbox(...)`` / ``label._resize_bbox(...)`` call sites (and the
existing tests) keep resolving to the very same function objects. These identity
assertions fail loudly if an alias is ever renamed, dropped, or re-implemented
instead of delegating."""

from digitalsreeni_image_annotator.widgets import edit_gestures
from digitalsreeni_image_annotator.widgets.image_label import ImageLabel


def test_static_geometry_helpers_are_aliases_to_edit_gestures():
    assert ImageLabel._bbox_handle_points is edit_gestures.bbox_handle_points
    assert ImageLabel._resize_bbox is edit_gestures.resize_bbox
    assert ImageLabel._scale_segmentation is edit_gestures.scale_segmentation
    assert ImageLabel._translate_segmentation is edit_gestures.translate_segmentation
    assert ImageLabel._scale_keypoints is edit_gestures.scale_keypoints
    assert ImageLabel._translate_keypoints is edit_gestures.translate_keypoints
    assert ImageLabel._sync_bbox_key is edit_gestures.sync_bbox_key


def test_selection_color_is_re_exported_from_canvas_renderer():
    from digitalsreeni_image_annotator.widgets.canvas_renderer import CanvasRenderer

    assert ImageLabel._SELECTION_COLOR is CanvasRenderer._SELECTION_COLOR
