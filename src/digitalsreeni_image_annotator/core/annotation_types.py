"""Typed shapes for the annotation data model (issue #78).

The codebase's most meaning-carrying structure is a dict whose valid shapes
were documented in prose. The sharpest example: a pose instance is
distinguished from a polygon by the **absence** of a key (ADR-029) — a rule
that a type checker can enforce and that until now only a comment protected.

These are ``TypedDict``s with ``total=False``, deliberately. The annotation
dict genuinely varies and genuinely gains keys at runtime: ``segmentation_raw``
appears lazily the first time a mask is thinned (ADR-025), ``source`` and
``track_run`` land on tracked results (ADR-040), ``assigned_class`` on
unprompted proposals (#69). Forcing a rigid schema onto data that is
legitimately open would produce false errors and teach people to ignore the
checker.

Qt-free, so both the GUI and the CLI can import them.
"""

from collections import Counter
from typing import TypedDict

# --- aliases for the recurring shapes --------------------------------------

#: Flat ``[x1, y1, x2, y2, ...]`` polygon ring. Unclosed by convention: the
#: first vertex is *not* repeated at the end (shapely's closing vertex is
#: stripped wherever one is produced).
Polygon = list[float]

#: ``[x, y, width, height]``. Note this is NOT ``[xmin, ymin, xmax, ymax]`` --
#: Pascal VOC uses corners and the importer converts (issue #75).
BBox = list[float]

#: Flat ``[x1, y1, v1, x2, y2, v2, ...]``. ``v`` is 0 not-labelled, 1 occluded,
#: 2 visible. A v=0 point is padded at ``(0, 0)`` and is not a coordinate.
Keypoints = list[float]

#: ``{class_name: class_id}``.
ClassMapping = dict[str, int]


def resolve_category_id(
    class_mapping: ClassMapping,
    class_name: str,
    skipped: Counter | None = None,
) -> int | None:
    """Category id for ``class_name``, or ``None`` if the project has no such class.

    Shared by the three commit loops (DINO auto-accept, review accept, SAM 3
    tracking). Passing a ``Counter`` as ``skipped`` tallies the losses by class
    name so the caller can report them -- two of the three used to drop the
    annotation with only a ``logger.warning`` and still report success.

    The loops themselves stay separate: they differ in shape, and
    ``accept_dino_results`` must NOT pre-assign ``number`` because
    ``add_annotation_to_list`` derives it after the append.
    """
    category_id = class_mapping.get(class_name)
    if category_id is None and skipped is not None:
        skipped[class_name] += 1
    return category_id


class KeypointSchema(TypedDict, total=False):
    """Per-class pose schema (ADR-029). One per class, the COCO rule.

    ``skeleton`` is 0-based here and in the ``.iap`` file; COCO's own
    ``skeleton`` is 1-based and the exporter converts. ``flip_idx`` is an
    app-level extension kept 0-based in both.
    """

    names: list[str]
    skeleton: list[list[int]]
    flip_idx: list[int]


class PolygonAnnotation(TypedDict, total=False):
    """A mask. Carries ``segmentation``; may also carry a derived ``bbox``."""

    segmentation: Polygon
    bbox: BBox
    category_id: int
    category_name: str
    number: int
    type: str
    #: Lazily captured full-precision copy, the source Detail-% re-simplifies
    #: from (ADR-025). Absent until a mask is first thinned, and invalidated by
    #: a vertex-count change (#68).
    segmentation_raw: Polygon
    detail_pct: int
    score: float
    source: str


class BBoxAnnotation(TypedDict, total=False):
    """A box-only annotation, as produced by a detection import.

    Beware: these carry ``"segmentation": None`` when built by some import
    paths, which is why existence-only ``"segmentation" in ann`` checks are a
    hazard and truthiness is the safe test.
    """

    bbox: BBox
    category_id: int
    category_name: str
    number: int
    type: str
    score: float
    source: str


class PoseAnnotation(TypedDict, total=False):
    """A pose instance.

    **There is deliberately no ``segmentation`` key in this definition.** Its
    absence is the discriminator the whole app routes on — area calculation,
    the Detail-% column, the ``draw_annotations`` branch order, merge and
    change-class guards (ADR-029). Writing one, even as ``None``, breaks every
    existence-only check that is not None-guarded.
    """

    keypoints: Keypoints
    num_keypoints: int
    bbox: BBox
    category_id: int
    category_name: str
    number: int
    score: float
    source: str


#: Any annotation. Discriminate with :func:`is_pose` / :func:`is_polygon`
#: rather than by key membership, for the reasons above.
Annotation = PolygonAnnotation | BBoxAnnotation | PoseAnnotation

#: ``{class_name: [annotation, ...]}`` for one image.
AnnotationsByClass = dict[str, list[Annotation]]

#: ``{image_or_slice_name: {class_name: [annotation, ...]}}``. Keyed by image
#: **and** slice name, which is why iterating it covers slices for free and
#: iterating ``all_images`` instead silently skips them.
AnnotationsByImage = dict[str, AnnotationsByClass]


def is_pose(annotation: Annotation) -> bool:
    """True for a pose instance.

    The canonical discriminator, expressed once: has keypoints and has no
    usable segmentation. Truthiness rather than ``in``, because a bbox-only
    import can carry ``"segmentation": None``.
    """
    return "keypoints" in annotation and not annotation.get("segmentation")


def is_polygon(annotation: Annotation) -> bool:
    """True when the annotation has a usable mask outline."""
    return bool(annotation.get("segmentation"))


def is_bbox_only(annotation: Annotation) -> bool:
    """True for a box with neither a mask nor keypoints."""
    return (
        not is_pose(annotation)
        and not is_polygon(annotation)
        and bool(annotation.get("bbox"))
    )
