"""
CanvasContext — narrow read-only view of main-window state used by
ImageLabel during rendering and event handling.

The orchestrator (ImageAnnotator) constructs one CanvasContext and
passes it to ImageLabel via set_context(). ImageLabel reads state
through the accessors here; writes go out as Qt signals connected to
controllers in ImageAnnotator.__init__.

All accessors are methods (not attributes) so the source of truth
stays on ImageAnnotator and future refactors that move state to a
controller can re-route the accessor without changing ImageLabel.
"""


class CanvasContext:
    def __init__(self, main_window):
        self._mw = main_window

    def paint_brush_size(self) -> int:
        return self._mw.paint_brush_size

    def eraser_size(self) -> int:
        return self._mw.eraser_size

    def current_class(self):
        return self._mw.current_class

    def class_id(self, name: str) -> int:
        return self._mw.class_mapping[name]

    def class_mapping(self) -> dict:
        return self._mw.class_mapping

    def is_class_visible(self, name: str) -> bool:
        return self._mw.class_controller.is_class_visible(name)

    def keypoint_schema(self, name):
        """Keypoint schema for a pose class, or None for a normal class
        (issue #35). value: {"names", "skeleton", "flip_idx"}."""
        return self._mw.keypoint_schemas.get(name)

    def current_image_key(self):
        return self._mw.current_slice or self._mw.image_file_name

    def has_annotation_selection(self) -> bool:
        # The annotation list is the source of truth for what a Delete acts
        # on; the canvas Delete must read it (not the possibly-stale red
        # highlight) so it can't fire on an empty list selection. See ADR-022.
        return bool(self._mw.annotation_list.selectedItems())

    def all_annotations(self) -> dict:
        return self._mw.all_annotations

    def scroll_area(self):
        return self._mw.scroll_area

    def dialog_parent(self):
        return self._mw
