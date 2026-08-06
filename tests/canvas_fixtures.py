"""Shared canvas-test doubles for the ImageLabel interaction tests (issue #77).

The canvas layer had almost no coverage before this module existed: the tool
handlers sat at 22-27 %, ``canvas_renderer`` at 49 %. Every one of those paths
is reachable only through a mouse or key event, so testing them needs three
things that are tedious to rebuild per test file:

* :class:`FakeCanvasContext` -- ``CanvasContext`` is already the narrow read
  accessor (ADR-018), so a fake is cheap and keeps the tests free of a main
  window.
* :class:`FakeMouseEvent` -- the handlers read only ``button()``,
  ``buttons()`` and ``modifiers()``.
* :class:`RecordingPainter` -- records the ``QPainter`` call sequence so the
  renderer's *contract* (draw order, class-visibility filtering) can be
  asserted without comparing pixels.

Imported by ``tests/unit/test_tool_handlers.py``,
``tests/unit/test_canvas_renderer_contract.py`` and
``tests/ui/test_canvas_gestures.py``. Kept out of ``conftest.py`` because these
are classes, not fixtures -- the fixtures that build on them live next to the
tests that use them.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap


class FakeCanvasContext:
    """Stand-in for :class:`CanvasContext` (ADR-018).

    Only the accessors the canvas actually calls are implemented; anything the
    code under test reaches for that is missing here is a real coupling the
    test should surface as an ``AttributeError`` rather than silently absorb.
    """

    def __init__(
        self,
        current_class="cell",
        classes=("cell",),
        hidden=(),
        schemas=None,
        image_key="img.png",
        paint_size=5,
        eraser_size=5,
    ):
        self._current_class = current_class
        self._classes = list(classes)
        self._hidden = set(hidden)
        self._schemas = dict(schemas or {})
        self._image_key = image_key
        self._paint_size = paint_size
        self._eraser_size = eraser_size
        self.all_annotations_dict = {}

    def paint_brush_size(self):
        return self._paint_size

    def eraser_size(self):
        return self._eraser_size

    def current_class(self):
        return self._current_class

    def class_id(self, name):
        return self._classes.index(name) + 1 if name in self._classes else 0

    def class_mapping(self):
        return {name: i + 1 for i, name in enumerate(self._classes)}

    def is_class_visible(self, name):
        return name not in self._hidden

    def keypoint_schema(self, name):
        return self._schemas.get(name)

    def current_image_key(self):
        return self._image_key

    def has_annotation_selection(self):
        return False

    def all_annotations(self):
        return self.all_annotations_dict

    def scroll_area(self):
        return None

    def dialog_parent(self):
        return None


class FakeMouseEvent:
    """Minimal ``QMouseEvent`` stand-in for the tool handlers.

    ``buttons()`` (plural, the *held* buttons during a move) defaults to match
    ``button()`` so a drag reads as held-down without the caller spelling it
    out; pass ``buttons=Qt.MouseButton.NoButton`` for a bare hover.
    """

    def __init__(self, button=Qt.MouseButton.LeftButton, modifiers=None, buttons=None):
        self._button = button
        self._buttons = button if buttons is None else buttons
        self._modifiers = modifiers or Qt.KeyboardModifier.NoModifier

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons

    def modifiers(self):
        return self._modifiers


class RecordingPainter:
    """Records every ``QPainter`` call as ``(name, args)`` instead of painting.

    ``CanvasRenderer`` only ever *calls* the painter -- it never reads a value
    back -- so a recorder is a faithful substitute and lets a test assert the
    draw contract (order of layers, which shapes were drawn at all) without
    rendering pixels or comparing images.
    """

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args))
            return None

        return _record

    def names(self):
        """Just the method names, in call order."""
        return [name for name, _ in self.calls]

    def texts(self):
        """Every string passed to ``drawText`` (the annotation labels)."""
        out = []
        for name, args in self.calls:
            if name == "drawText":
                out.extend(a for a in args if isinstance(a, str))
        return out

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)

    def index_of(self, name, occurrence=0):
        """Call-order index of the ``occurrence``-th ``name`` call, or -1."""
        seen = 0
        for i, (n, _) in enumerate(self.calls):
            if n == name:
                if seen == occurrence:
                    return i
                seen += 1
        return -1


def make_label(qtbot, width=200, height=200, ctx=None, zoom=1.0):
    """A real, model-less :class:`ImageLabel` with a solid pixmap loaded.

    ``original_pixmap`` must be set for the mouse handlers to do anything at
    all (every one of them early-returns without it), and the paint/eraser
    tools read its size to allocate their mask.
    """
    from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel

    label = ImageLabel(None)
    qtbot.addWidget(label)
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#404040"))
    label.zoom_factor = zoom
    label.ui_scale = 1.0
    label.setPixmap(pixmap)
    # setPixmap -> update_offset centres the image in the widget, which is 0x0
    # until it is shown. Pin the offset to 0 so image coords == widget coords
    # and the tests can express positions in image space.
    label.offset_x = 0
    label.offset_y = 0
    label.set_context(ctx or FakeCanvasContext())
    return label


def square(x0, y0, side, name="cell", number=1):
    """A polygon annotation: axis-aligned square as a flat coordinate ring."""
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
        "number": number,
    }


def bbox(x, y, w, h, name="cell", number=1):
    """A box-only annotation (what an import without masks produces)."""
    return {"bbox": [x, y, w, h], "category_name": name, "number": number}


def pose(points, name="person", number=1):
    """A pose instance: ``points`` is ``[(x, y, v), ...]``.

    Deliberately carries **no** ``segmentation`` key -- that absence is the
    discriminator routing area, Detail-% and rendering (ADR-029), so a test
    fixture that added one would be testing a shape the app never produces.
    """
    flat = [c for p in points for c in p]
    xs = [p[0] for p in points if p[2] > 0]
    ys = [p[1] for p in points if p[2] > 0]
    return {
        "keypoints": flat,
        "num_keypoints": sum(1 for p in points if p[2] > 0),
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "category_name": name,
        "number": number,
    }
