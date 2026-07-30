"""CanvasRenderer contract tests (issue #77, ADR-034).

The renderer is asserted through a :class:`RecordingPainter` rather than by
comparing pixels: what matters is the **contract** -- which layers are drawn,
in what order, and which annotations are filtered out -- not the exact
rasterisation, which legitimately changes with antialiasing, Qt version and
DPI.

This is the harness the onion-skinning work (issue #67) builds on: an
additional layer is only safe to insert if the existing layer order is pinned
by a test.
"""

import pytest
from PyQt6.QtGui import QColor

from tests.canvas_fixtures import (
    FakeCanvasContext,
    RecordingPainter,
    bbox,
    make_label,
    pose,
    square,
)


@pytest.fixture
def label(qtbot):
    lbl = make_label(qtbot)
    lbl.class_colors = {"cell": QColor("#1F77B4"), "debris": QColor("#FF7F0E"),
                        "person": QColor("#2CA02C")}
    return lbl


# --- class-visibility filtering --------------------------------------------


def test_hidden_class_is_not_drawn(label):
    label._ctx = FakeCanvasContext(hidden={"debris"}, classes=("cell", "debris"))
    label.annotations = {
        "cell": [square(10, 10, 40)],
        "debris": [square(100, 100, 40)],
    }
    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)

    assert painter.count("drawPolygon") == 1, "only the visible class draws"
    assert painter.texts() == ["cell 1"]


def test_all_classes_hidden_draws_no_shape(label):
    label._ctx = FakeCanvasContext(hidden={"cell"}, classes=("cell",))
    label.annotations = {"cell": [square(10, 10, 40)]}
    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)
    assert painter.count("drawPolygon") == 0


def test_selection_chrome_is_suppressed_over_a_hidden_mask(label):
    """A selected-but-hidden mask must not leak its position through the
    selection handles."""
    label._ctx = FakeCanvasContext(hidden={"cell"}, classes=("cell",))
    annotation = square(10, 10, 40)
    label.annotations = {"cell": [annotation]}
    label.highlighted_annotations = [annotation]
    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)
    assert painter.count("drawRect") == 0


# --- draw order ------------------------------------------------------------


def test_selection_overlay_is_drawn_after_every_mask(label):
    """The overlay is a final pass so it sits on top of *all* fills, including
    a later-drawn overlapping mask (ADR-022 amendment)."""
    first = square(10, 10, 40, name="cell")
    second = square(20, 20, 40, name="debris")
    label.annotations = {"cell": [first], "debris": [second]}
    label.highlighted_annotations = [first]

    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)

    last_polygon = max(
        i for i, (n, _) in enumerate(painter.calls) if n == "drawPolygon"
    )
    first_overlay_rect = painter.index_of("drawRect")
    assert first_overlay_rect > last_polygon


def test_paint_event_layer_order(qtbot):
    """End-to-end layer order through the real ``paintEvent``.

    Asserted by monkeypatching the renderer methods to append a marker: the
    image, then committed annotations, then tool overlays, then the temp
    (review) annotations on top. Onion-skinning inserts *before* the image, so
    this ordering is the invariant it must not disturb.
    """
    label = make_label(qtbot)
    label._ctx = FakeCanvasContext()
    label.annotations = {"cell": [square(10, 10, 40)]}
    label.temp_annotations = [
        {"segmentation": [5, 5, 15, 5, 15, 15], "category_name": "Temp-x",
         "score": 0.9, "source": "dino"}
    ]

    order = []
    for name in ("draw_annotations", "draw_tool_size_indicator", "draw_temp_annotations"):
        setattr(label.renderer, name, (lambda n: lambda *a, **k: order.append(n))(name))

    label.grab()  # force exactly one real paintEvent

    assert order == [
        "draw_annotations",
        "draw_tool_size_indicator",
        "draw_temp_annotations",
    ]


# --- per-shape branching ---------------------------------------------------


def test_pose_instance_takes_the_keypoint_branch_not_the_bbox_branch(label):
    """A pose carries a bbox too; the ``elif "keypoints"`` branch sits *before*
    ``elif "bbox"`` so the skeleton wins (ADR-029)."""
    label._ctx = FakeCanvasContext(
        classes=("person",),
        schemas={"person": {"names": ["a", "b"], "skeleton": [[0, 1]], "flip_idx": [0, 1]}},
    )
    label.annotations = {"person": [pose([(10, 10, 2), (30, 30, 2)])]}

    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)

    assert painter.count("drawLine") == 1, "skeleton edge drawn"
    assert painter.count("drawEllipse") == 2, "one marker per labelled point"


def test_unlabelled_keypoints_are_not_drawn(label):
    label._ctx = FakeCanvasContext(
        classes=("person",),
        schemas={"person": {"names": ["a", "b"], "skeleton": [[0, 1]], "flip_idx": [0, 1]}},
    )
    label.annotations = {"person": [pose([(10, 10, 2), (0, 0, 0)])]}

    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)

    assert painter.count("drawEllipse") == 1
    assert painter.count("drawLine") == 0, "skeleton edge needs both ends labelled"


def test_bbox_only_annotation_draws_a_rect(label):
    label.annotations = {"cell": [bbox(10, 10, 40, 40)]}
    painter = RecordingPainter()
    label.renderer.draw_annotations(painter)
    assert painter.count("drawRect") == 1
    assert painter.count("drawPolygon") == 0


# --- temp (review) annotations ---------------------------------------------


def test_temp_annotation_prefers_the_polygon_over_the_bbox(label):
    """DINO+SAM temp annotations carry both; the mask is the useful one."""
    label.temp_annotations = [{
        "segmentation": [10, 10, 40, 10, 40, 40],
        "bbox": [10, 10, 30, 30],
        "category_name": "Temp-cell",
        "score": 0.87,
    }]
    painter = RecordingPainter()
    label.renderer.draw_temp_annotations(painter)

    assert painter.count("drawPolygon") == 1
    assert painter.count("drawRect") == 0
    assert painter.texts() == ["Temp-cell 0.87"]


def test_temp_annotation_falls_back_to_the_bbox(label):
    label.temp_annotations = [
        {"bbox": [10, 10, 30, 30], "category_name": "Temp-cell", "score": 0.5}
    ]
    painter = RecordingPainter()
    label.renderer.draw_temp_annotations(painter)
    assert painter.count("drawRect") == 1


# --- painter helpers -------------------------------------------------------


def test_pen_width_is_zoom_compensated(label):
    """Overlay chrome keeps a constant on-screen width across zoom."""
    label.zoom_factor = 1.0
    assert label.renderer._pen_w(2) == pytest.approx(2.0)
    label.zoom_factor = 4.0
    assert label.renderer._pen_w(2) == pytest.approx(0.5)


def test_pen_width_scales_with_the_ui_font(label):
    label.zoom_factor = 1.0
    label.ui_scale = 2.0
    assert label.renderer._pen_w(2) == pytest.approx(4.0)


def test_overlay_font_never_goes_below_one_point(label):
    label.zoom_factor = 1000.0
    assert label.renderer._overlay_font(12).pointSize() >= 1
