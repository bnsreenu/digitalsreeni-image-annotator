"""Screen <-> image coordinate conversion (issue #77).

``get_image_coordinates`` is the single funnel every mouse event passes through
before it reaches a tool handler, so a sign error or a forgotten ``offset``
term here miscomputes *every* gesture in the app at once. It had no direct test.

The reference-frame rules this pins down are the ones documented in
"Pan + Zoom Reference Frames": pan is tracked in **global** coords, and the
post-zoom offset is derived from the **viewport** size, not ``self.width()``.
Both exist because the widget-local frame shifts underneath the cursor.
"""

import pytest
from PyQt6.QtCore import QPointF

from tests.canvas_fixtures import make_label


@pytest.fixture
def label(qtbot):
    return make_label(qtbot, width=400, height=300)


def test_identity_at_unit_zoom_and_zero_offset(label):
    assert label.get_image_coordinates(QPointF(37, 91)) == (37, 91)


@pytest.mark.parametrize("zoom", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_zoom_divides_out(label, zoom):
    label.zoom_factor = zoom
    assert label.get_image_coordinates(QPointF(100 * zoom, 60 * zoom)) == (100, 60)


@pytest.mark.parametrize("offset", [(0, 0), (25, 0), (0, 40), (25, 40)])
def test_offset_subtracts_before_the_zoom_divide(label, offset):
    """Offset is a screen-space translation, so it comes off *first*."""
    label.offset_x, label.offset_y = offset
    label.zoom_factor = 2.0
    screen = QPointF(offset[0] + 2 * 30, offset[1] + 2 * 45)
    assert label.get_image_coordinates(screen) == (30, 45)


def test_conversion_truncates_towards_zero(label):
    """int() truncation, not rounding -- a pixel is the cell you are inside."""
    label.zoom_factor = 3.0
    assert label.get_image_coordinates(QPointF(8, 8)) == (2, 2)  # 8/3 = 2.66


def test_negative_screen_positions_stay_negative(label):
    """Dragging past the top-left must not wrap; the clamp happens at commit
    (ADR-024), not here."""
    label.offset_x, label.offset_y = 50, 50
    assert label.get_image_coordinates(QPointF(10, 10)) == (-40, -40)


def test_no_pixmap_returns_origin(qtbot):
    from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel

    bare = ImageLabel(None)
    qtbot.addWidget(bare)
    assert bare.get_image_coordinates(QPointF(123, 456)) == (0, 0)


def test_roundtrip_through_the_forward_transform(label):
    """The renderer's forward transform is translate(offset) then scale(zoom);
    the inverse here must undo exactly that."""
    label.zoom_factor = 2.5
    label.offset_x, label.offset_y = 17, 23
    for image_point in [(0, 0), (10, 10), (199, 149)]:
        screen = QPointF(
            image_point[0] * label.zoom_factor + label.offset_x,
            image_point[1] * label.zoom_factor + label.offset_y,
        )
        assert label.get_image_coordinates(screen) == image_point


def test_update_offset_centres_the_scaled_pixmap(label):
    """The centring offset is what makes a zoomed-out image sit in the middle
    rather than hugging the top-left."""
    label.resize(600, 500)
    label.zoom_factor = 1.0
    label.update_scaled_pixmap()
    assert label.offset_x == (600 - 400) // 2
    assert label.offset_y == (500 - 300) // 2
