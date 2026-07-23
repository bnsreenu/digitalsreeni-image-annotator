"""
Keypoint (pose) schema helpers — issue #35.

A *pose class* carries a keypoint schema describing its ordered, named
keypoints and the skeleton edges between them (the COCO instance model:
every instance of the class shares one schema). The schema is a plain,
JSON-able dict so it round-trips through ``.iap`` projects via
``image_utils.convert_to_serializable`` with no special handling:

    {
        "names":    ["nose", "left_eye", "right_eye", ...],  # ordered, length K
        "skeleton": [[0, 1], [0, 2], ...],                   # 0-based index edges
        "flip_idx": [0, 2, 1, ...],                          # length K, h-flip map
    }

These helpers are Qt-free and pure so they can be unit-tested without a
QApplication and reused by the schema dialog, the COCO/YOLO importers, and
project load. ``flip_idx[i]`` is the index point *i* maps to under a
horizontal flip (identity for points on the symmetry axis); it is required
by YOLO-pose augmentation and app-only for COCO.
"""


def make_schema(names, skeleton=None, flip_idx=None):
    """Build a normalized schema dict from parts.

    ``flip_idx`` defaults to the identity permutation (no left/right swap).
    Returns the sanitized dict; raises ValueError if the parts are invalid.
    """
    schema = sanitize_schema(
        {
            "names": list(names),
            "skeleton": [list(edge) for edge in (skeleton or [])],
            "flip_idx": list(flip_idx) if flip_idx is not None else list(range(len(names))),
        }
    )
    if schema is None:
        raise ValueError("Invalid keypoint schema")
    return schema


def sanitize_schema(schema):
    """Validate and normalize a (possibly user- or file-supplied) schema.

    Returns a clean dict ``{"names", "skeleton", "flip_idx"}`` or ``None`` if
    the input is not a usable schema. Never raises — callers can treat ``None``
    as "not a pose class / drop it".

    Rules: names must be a non-empty list of non-empty, unique strings;
    skeleton edges must be in-range 0-based index pairs (out-of-range or
    malformed edges are dropped); flip_idx must be a length-K permutation of
    0..K-1 (anything else falls back to identity).
    """
    if not isinstance(schema, dict):
        return None

    names = schema.get("names")
    if not isinstance(names, (list, tuple)) or not names:
        return None
    names = [str(n) for n in names]
    if any(not n.strip() for n in names):
        return None
    if len(set(names)) != len(names):
        return None
    k = len(names)

    skeleton = []
    raw_skeleton = schema.get("skeleton") or []
    if isinstance(raw_skeleton, (list, tuple)):
        for edge in raw_skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            try:
                a, b = int(edge[0]), int(edge[1])
            except (TypeError, ValueError):
                continue
            if 0 <= a < k and 0 <= b < k and a != b:
                skeleton.append([a, b])

    flip_idx = _sanitize_flip_idx(schema.get("flip_idx"), k)

    return {"names": names, "skeleton": skeleton, "flip_idx": flip_idx}


def is_involution(flip_idx):
    """True iff `flip_idx` is a self-inverse permutation of 0..len-1 — the
    shape a horizontal-flip mapping must have (flipping twice returns the
    original point). A permutation that isn't self-inverse (e.g. a 3-cycle)
    is nonsensical as a flip map even though it's a valid bijection."""
    k = len(flip_idx)
    if sorted(flip_idx) != list(range(k)):
        return False  # not even a permutation (duplicate or out-of-range target)
    return all(flip_idx[flip_idx[i]] == i for i in range(k))


def _sanitize_flip_idx(flip_idx, k):
    """A length-K self-inverse permutation of 0..K-1, or the identity if
    invalid (including a permutation that isn't self-inverse)."""
    identity = list(range(k))
    if not isinstance(flip_idx, (list, tuple)) or len(flip_idx) != k:
        return identity
    try:
        values = [int(v) for v in flip_idx]
    except (TypeError, ValueError):
        return identity
    if not is_involution(values):
        return identity
    return values


def schema_k(schema):
    """Number of keypoints K in a schema (0 if not a valid schema)."""
    if not isinstance(schema, dict):
        return 0
    names = schema.get("names")
    return len(names) if isinstance(names, (list, tuple)) else 0
