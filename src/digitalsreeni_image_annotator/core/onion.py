"""Onion-skin neighbour resolution (issue #67).

Stepping through a Z-stack or a video frame by frame gave no visual reference
to the neighbouring slice: you could not see how far an object had moved,
whether you were drifting, or whether you had already annotated the equivalent
structure one slice back. Every slice was annotated blind.

This module answers only one question — *which* slices are the neighbours — and
answers it with plain strings and integers so it can be tested without a Qt
application or a decoded image anywhere in sight. Materialising those slices is
the caller's job, and goes through the existing bounded LRU (ADR-036) so the
ghost costs no extra ownership of decoded pixels.

Qt-free on purpose: neighbour selection is index arithmetic, and index
arithmetic that silently wraps at the ends of a stack is a bug that should be
caught by a unit test, not by a user noticing the last frame ghosting the first.
"""

from collections.abc import Iterable
from typing import Any

# `Any` rather than a narrower type on the clamps below is deliberate: their
# input is whatever QSettings returned, which is genuinely untyped (it
# round-trips values as strings on some backends and can hold anything a
# hand-edited registry or INI contains). That is exactly why they clamp.

# Which neighbours to show. "previous" is the default because the common
# workflow is stepping forward through a stack while checking against what you
# just annotated.
MODE_PREVIOUS = "previous"
MODE_NEXT = "next"
MODE_BOTH = "both"
MODES = (MODE_PREVIOUS, MODE_NEXT, MODE_BOTH)

DEFAULT_MODE = MODE_PREVIOUS
DEFAULT_OFFSET = 1
# Tuned for the DEFAULT content, which is a 2 px dashed outline rather than a
# blended raster: 0.35 was chosen for the wash and leaves an outline too faint
# to line anything up against. The slider still spans 0.05..0.95 for whoever
# turns the image ghost on.
DEFAULT_OPACITY = 0.55

# WHAT gets ghosted, which turns out to matter more than which neighbour does.
# Ghosting the neighbour's *pixels* is the animator's onion skin, and on an
# opaque photographic slice it mostly reads as "this image is out of focus" --
# it was the first thing tried and the first thing complained about. The
# question actually worth answering while stepping through a stack is "what did
# I label on the previous slice, and does this one line up with it?", so the
# ANNOTATION ghost is the default and the raster ghost is opt-in.
CONTENT_ANNOTATIONS = "annotations"
CONTENT_IMAGE = "image"
CONTENT_BOTH = "both"
CONTENTS = (CONTENT_ANNOTATIONS, CONTENT_IMAGE, CONTENT_BOTH)

DEFAULT_CONTENT = CONTENT_ANNOTATIONS

# Offsets beyond this are not useful (the ghost is unrecognisable) and each one
# is another live decode competing for the shared LRU's 8 slots.
MAX_OFFSET = 5


def clamp_opacity(value: Any) -> float:
    """Coerce a stored/passed opacity into 0.05..0.95.

    Never 0 (indistinguishable from "off", but with the decode cost still paid)
    and never 1 (the ghost would completely hide the current slice).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_OPACITY
    return max(0.05, min(0.95, value))


def clamp_offset(value: Any) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_OFFSET
    return max(1, min(MAX_OFFSET, value))


def normalise_mode(value: Any) -> str:
    return value if value in MODES else DEFAULT_MODE


def normalise_content(value: Any) -> str:
    return value if value in CONTENTS else DEFAULT_CONTENT


def wants_annotations(content: Any) -> bool:
    return normalise_content(content) in (CONTENT_ANNOTATIONS, CONTENT_BOTH)


def wants_image(content: Any) -> bool:
    """True when the neighbouring *pixels* are needed.

    Worth its own predicate because it gates the only expensive half of the
    feature: a False here skips a slice decode and the LRU traffic that comes
    with it, which is why the default content setting costs nothing but a dict
    lookup per neighbour.
    """
    return normalise_content(content) in (CONTENT_IMAGE, CONTENT_BOTH)


def neighbour_names(
    names: Iterable[str] | None,
    current: str | None,
    offset: int = DEFAULT_OFFSET,
    mode: str = DEFAULT_MODE,
) -> list[str]:
    """Names of the onion-skin neighbours of ``current`` within ``names``.

    Returns them in draw order (earlier first). Out-of-range neighbours are
    **omitted, never wrapped**: at the first slice there is no previous one,
    and ghosting the last slice there would be actively misleading. An unknown
    ``current`` or a collection with fewer than two entries yields ``[]``, which
    is how a single image ends up with no ghost without a special case.
    """
    names = list(names or [])
    if len(names) < 2 or current not in names:
        return []

    index = names.index(current)
    offset = clamp_offset(offset)
    mode = normalise_mode(mode)

    wanted = []
    if mode in (MODE_PREVIOUS, MODE_BOTH):
        wanted.append(index - offset)
    if mode in (MODE_NEXT, MODE_BOTH):
        wanted.append(index + offset)

    return [names[i] for i in wanted if 0 <= i < len(names)]


def is_available(names: Iterable[str] | None) -> bool:
    """True when a collection has enough slices for onion-skinning to mean
    anything. Drives whether the controls are enabled at all."""
    return len(list(names or [])) > 1
