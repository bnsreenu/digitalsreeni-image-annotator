"""Group-aware train/val splitting (issue #81, ADR-044).

The split used to be keyed by the **image name**. That is correct only when
every name is an independent observation, and in this app it routinely is not:
a multi-dimensional stack contributes one name per slice (``stack_T1_Z5_C1``)
and a video one per frame (``video_F00042``). Consecutive frames of one
recording are near-identical, so a name-keyed split scatters them across train
and val by construction -- the model is validated on frames it effectively
trained on, and every reported validation metric comes back optimistic. The
numbers look better the more redundant the data is, which is precisely
backwards.

So the split key is the **group**, not the name. A group is "one source of
observations", and the whole group lands on one side.

**Groups are derived from structure, not from a model.** The primary source is
``image_slices``, already keyed by the ext-stripped base name, so the mapping is
exact and free. Near-duplicate clusters from the curation feature (#72) refine
it further through :func:`merge_groups`, but they are never *required* and
nothing here waits for them: the worst leakage -- a 200-frame video -- is fixed
without a model, a GPU or a curation run. The refinement closes the case the
names cannot see: a folder of frames extracted as real files
(``clip_F00001.png``), where the dot says "independent file" and the pixels say
otherwise.

Qt-free (ADR-041), and deliberately importing nothing from ``slice_cache``:
that module reaches ``core.image_utils``, which imports ``QImage``. The
three-line ``.names`` accessor is inlined below for the same reason
``core.slice_index`` inlines it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

# A slice name is the ext-stripped base plus one `<DimLetter><1-based index>`
# component per non-spatial dimension (`SliceProvider._build_index`), or
# `_F#####` for a video frame (`video_handler.frame_key`, ADR-037). Regular
# image names keep their extension, which is why the dot check below tells the
# two apart -- the same signal the exporters already use.
#
# The letters are exactly the ones `DimensionDialog` offers minus the spatial
# pair (H and W never become slice components), plus F for video. Matching any
# `[A-Z]` instead looked more permissive but was actively wrong: a 96-well
# plate exported as `Plate1_A1_T1_Z1` would have been grouped under `Plate1`,
# collapsing 96 independent stacks into one group -- which degrades to the
# per-name split this module exists to replace, while reporting that everything
# came from one recording.
_SLICE_SUFFIX = re.compile(r"^(?P<base>.+?)(?:_[TZCSF]\d+)+$")


def _collection_names(collection: Any) -> list[str]:
    """Slice names of a collection, decoding nothing.

    ``.names`` on a ``LazySliceList``; a plain ``[(name, qimage), ...]``
    otherwise, which legacy call sites and several tests still hand in.
    """
    names = getattr(collection, "names", None)
    if names is not None:
        return list(names)
    return [name for name, _ in (collection or [])]


def _slice_base(name: str) -> str | None:
    """The stack/video base name a slice name belongs to, or ``None``.

    A best-effort fallback for names with no ``image_slices`` entry -- the CLI
    passes an empty mapping (``cli.commands._export_dispatch``) and an ``.iap``
    can carry slice names whose stack was never materialised in this session.

    Erring toward over-grouping is preferable *within limits* -- a stack
    literally named ``run_T1`` yields base ``run`` and merges with ``run_T2``,
    costing split granularity where the opposite error would reopen the leak.
    But the limit is real: collapse far enough and every name lands in one
    group, which falls back to the per-name split anyway. That is why the
    dimension letters are enumerated rather than matched as any capital.
    """
    if "." in name:
        return None
    match = _SLICE_SUFFIX.match(name)
    return match.group("base") if match else None


def derive_groups(
    names: Iterable[str], image_slices: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """``{name: group_key}`` for every name in ``names``.

    ``image_slices`` is the main window's ``{ext_stripped_base: collection}``
    mapping; it gives an exact answer with no parsing and no pixel work. Names
    it does not cover fall back to :func:`_slice_base`, and anything left is its
    own group -- a plain image is a group of one.
    """
    exact: dict[str, str] = {}
    for base, collection in (image_slices or {}).items():
        for slice_name in _collection_names(collection):
            exact[slice_name] = base

    groups: dict[str, str] = {}
    for name in names:
        groups[name] = exact.get(name) or _slice_base(name) or name
    return groups


def merge_groups(
    groups: Mapping[str, str], clusters: Iterable[Iterable[str]]
) -> dict[str, str]:
    """Fold near-duplicate clusters into a name-derived grouping.

    ``{a: A, b: B, c: C}`` plus the cluster ``[a, b]`` yields one group for
    ``a`` and ``b``; and because grouping is transitive, a cluster touching two
    existing groups merges *both* of them entirely -- if ``a``'s whole stack is
    near-identical to ``b``'s, splitting them was never safe either.

    This is what makes a curation run pay for itself beyond the report: it
    catches the redundancy the names cannot, above all a folder of video frames
    extracted as ordinary files, where nothing in the name says they came from
    one recording.

    Names not already in ``groups`` are ignored rather than added: the grouping
    describes one specific set of names about to be split, and a cluster
    computed over the whole project routinely mentions images that are not in
    it.
    """
    merged = dict(groups)

    def resolve(key: str) -> str:
        while parent.get(key, key) != key:
            key = parent[key]
        return key

    parent: dict[str, str] = {}
    for cluster in clusters:
        members = [name for name in cluster if name in merged]
        if len(members) < 2:
            continue
        roots = sorted({resolve(merged[name]) for name in members})
        target = roots[0]
        for root in roots[1:]:
            parent[root] = target

    for name, group in merged.items():
        merged[name] = resolve(group)
    return merged


def translate_clusters(
    clusters: Iterable[Iterable[str]], names_by_key: Mapping[str, str]
) -> list[list[str]]:
    """Rewrite clusters of image names as clusters of split keys.

    The SAM path splits on ``"{index}:{name}"`` keys rather than on names, so
    that two same-named sources stay distinct entries (``sam_dataset.split_keys``).
    Clusters arrive keyed by name, and handing them over untranslated would
    match nothing at all -- silently, since :func:`merge_groups` ignores names
    it does not recognise.
    """
    keys_by_name: dict[str, list[str]] = {}
    for key, name in names_by_key.items():
        keys_by_name.setdefault(name, []).append(key)
    translated = [
        [key for name in cluster for key in keys_by_name.get(name, [])]
        for cluster in clusters
    ]
    return [cluster for cluster in translated if len(cluster) > 1]


def _split_by_group(
    names: list[str], groups: Mapping[str, str], val_count: int
) -> tuple[set[str], set[str]]:
    """Route whole groups into val, landing as near ``val_count`` as possible.

    Groups are ordered by a stable MD5 of the group key -- the same device the
    name-keyed split used, so the result is reproducible across runs and
    machines (unlike ``hash()``, which is salted per process).

    Precondition: ``names`` holds no duplicates. Every caller satisfies it
    (dict keys, or ``"{index}:{name}"`` pairs), and ``plan_split`` guards on
    the distinct count — a repeat would be counted twice in a group's size
    while ``val_count`` came from the raw length.

    A group is indivisible, so ``val_count`` is a target rather than a
    guarantee, and choosing the best *set* is subset-sum. This settles for a
    local optimum instead: fill with groups that fit, allow one large group to
    replace the whole selection when that lands nearer, then hill-climb on
    single moves -- **add, drop or swap** -- until none improves.

    The guarantee is therefore: **no single group added, dropped, or swapped
    for another would land closer to the target.** That is exactly what the
    property test asserts, and it is the honest bound -- not that the requested
    percentage is delivered.

    The hill-climb enumerates one representative per distinct group *size*
    rather than every group. Swaps are otherwise quadratic in the group count,
    which matters: the ungrouped path runs this with one group per image, so
    a few thousand images would mean millions of pointless comparisons between
    interchangeable singletons.

    An earlier version added groups until it reached the target and then only
    reconsidered the last one. That is size-blind, and it delivered 1 % for a
    requested 20 % on a video-plus-one-photo project while every
    group-cohesion assertion stayed green.
    """
    members: dict[str, list[str]] = {}
    for name in names:
        members.setdefault(groups.get(name, name), []).append(name)

    ordered = sorted(
        members, key=lambda key: hashlib.md5(key.encode("utf-8")).hexdigest()
    )
    if len(ordered) < 2:
        # Callers route this to the ungrouped path; returning an empty val set
        # here is the one outcome the whole design says must not happen
        # quietly, so it is refused rather than left to a future caller.
        raise ValueError("a group split needs at least two groups")

    chosen: list[str] = []
    held_out = 0
    for key in ordered:
        if held_out + len(members[key]) <= val_count:
            chosen.append(key)
            held_out += len(members[key])
            if held_out == val_count:
                break

    def _distance(count: int) -> int:
        return abs(count - val_count)

    # Every group is larger than the target: hold out the smallest rather than
    # returning an empty val set. (Two groups minimum is a precondition above,
    # so this always leaves something in train.)
    if not chosen:
        chosen = [min(ordered, key=lambda key: (len(members[key]), key))]
        held_out = len(members[chosen[0]])

    # A single large group can beat the whole fill.
    best_single = min(ordered, key=lambda key: (_distance(len(members[key])), key))
    if _distance(len(members[best_single])) < _distance(held_out):
        chosen = [best_single]
        held_out = len(members[best_single])

    def _representatives(keys) -> dict[int, str]:
        """One canonical group per distinct size, first in hash order.

        Two groups of the same size are interchangeable for hitting a count,
        so only one of each needs considering. Picking the hash-first keeps the
        result deterministic.
        """
        best: dict[int, str] = {}
        for key in keys:
            best.setdefault(len(members[key]), key)
        return best

    # Hill-climb on single moves. Every accepted move strictly reduces a
    # non-negative integer distance, so this terminates; an exact hit short-
    # circuits it entirely, which is the common ungrouped case.
    selected = set(chosen)
    while _distance(held_out) > 0:
        outside = _representatives(key for key in ordered if key not in selected)
        inside = _representatives(key for key in ordered if key in selected)

        moves = []
        # The bounds keep both sides non-empty: an empty train set is not a
        # split, whatever the arithmetic says.
        if len(selected) + 1 < len(ordered):
            for size, key in outside.items():
                moves.append((_distance(held_out + size), "add", key, ""))
        if len(selected) > 1:
            for size, key in inside.items():
                moves.append((_distance(held_out - size), "drop", "", key))
        for in_size, in_key in outside.items():
            for out_size, out_key in inside.items():
                moves.append(
                    (_distance(held_out + in_size - out_size), "swap", in_key, out_key)
                )

        # `moves` is never empty: that would need a single group, which the
        # precondition above refuses.
        distance, kind, add_key, drop_key = min(moves)
        if distance >= _distance(held_out):
            break
        # Dispatch on the move kind, not on key truthiness: a group key that
        # happened to be the empty string would apply nothing, leave the
        # distance unchanged, and be re-selected forever.
        if kind in ("add", "swap"):
            selected.add(add_key)
            held_out += len(members[add_key])
        if kind in ("drop", "swap"):
            selected.discard(drop_key)
            held_out -= len(members[drop_key])

    val = {name for key in selected for name in members[key]}
    return set(names) - val, val


def plan_split(
    names: Iterable[str],
    val_pct: float,
    groups: Mapping[str, str] | None = None,
) -> tuple[set[str], set[str], bool]:
    """``(train, val, fell_back)`` for ``names`` at ``val_pct`` percent.

    ``fell_back`` is True when grouping was requested but could not be applied
    because everything belongs to a single group -- a project that is one video,
    typically. There the honest split does not exist: any val set shares a
    recording with train. Returning an empty val set would be the more truthful
    answer, but it makes the trainer silently skip validation and early stopping
    (ADR-028), which surfaces as a regression rather than as information. So the
    name-keyed split is used and the flag says so, leaving the UI to state
    plainly that the validation numbers will be optimistic.
    """
    ordered_names = list(names)
    total = len(ordered_names)
    # Distinct, not raw length: a duplicated name is one thing to place, and
    # two copies of it cannot be split. Guarding here is what makes the
    # two-group precondition in `_split_by_group` unreachable.
    if val_pct <= 0 or len(set(ordered_names)) < 2:
        return set(ordered_names), set(), False

    # Nearest integer, clamped so neither side is ever empty. round() is
    # half-to-even; the clamp makes that irrelevant at the boundaries.
    val_count = max(1, min(total - 1, round(total * val_pct / 100)))

    if not groups:
        return (*_split_by_group(ordered_names, {}, val_count), False)

    distinct = {groups.get(name, name) for name in ordered_names}
    if len(distinct) < 2:
        return (*_split_by_group(ordered_names, {}, val_count), True)

    return (*_split_by_group(ordered_names, groups, val_count), False)


def split_warning(
    names: Iterable[str],
    val_pct: float,
    image_slices: Mapping[str, Any] | None = None,
    groups: Mapping[str, str] | None = None,
) -> str | None:
    """What is wrong with the split about to happen, or ``None``.

    Lives here, not on the controller, because the CLI is a first-class path
    for this (ADR-044) and cannot import a module that pulls in Qt. Putting the
    wording next to the dialog meant the CLI hand-rolled a subset of it — the
    same duplication that let the preview and the export drift apart.

    Two conditions are worth telling someone about:

    * the grouping degenerates to one group, so no leak-free split exists;
    * the split leaves **training** with a single group. That is optimal by the
      image count the split aims at and useless as a dataset, and it is silent
      otherwise, because the grouping technically succeeded.

    ``groups`` overrides the derivation for callers that already know their own
    grouping — the SAM path keys by ``"{index}:{name}"``, and rebuilding an
    approximation of that here would preview a different split than the one
    that runs.
    """
    if val_pct <= 0:
        return None

    ordered = list(names)
    if groups is None:
        groups = derive_groups(ordered, image_slices)
    train, _val, fell_back = plan_split(ordered, val_pct, groups)

    if fell_back:
        return (
            "Every annotated image here falls into one group — one recording, "
            "or one set of similarly-named files — so no validation set can be "
            "held out without sharing near-identical images with training.\n\n"
            "The split was made per image instead. Near-identical images land "
            "on both sides, so the validation metrics this run reports will be "
            "optimistic: read them as a training-progress signal, not as an "
            "estimate of performance on new data.\n\n"
            "Data from a second recording would give a validation set that "
            "measures something."
        )

    # Strictly more than two groups, deliberately. With exactly two, training
    # holding one and validating on the other is not a degenerate split at all
    # -- it is the textbook one, and warning about it would fire on the
    # healthiest possible two-recording project.
    total_groups = len({groups.get(name, name) for name in ordered})
    train_groups = {groups.get(name, name) for name in train}
    if total_groups > 2 and len(train_groups) == 1:
        return (
            "This split leaves the training set with a single group and puts "
            "every other image into validation.\n\n"
            f"Groups are held out whole, so asking for {val_pct}% by image "
            "count consumed all of them but one. The model would train on one "
            "recording and be validated entirely on images unlike it — the "
            "mirror image of the problem the grouping exists to prevent.\n\n"
            "A different validation percentage, or more annotations outside "
            "the largest group, would give a split worth measuring."
        )
    return None


def assign_train_val(
    image_names: Iterable[str],
    val_pct: float,
    groups: Mapping[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    """Deterministically partition image names into ``(train, val)``.

    ``val_pct`` in ``[0, 100]``; 0 keeps everything in train. Without ``groups``
    this is the historical per-name split, unchanged. Re-exported from
    ``io.export_formats``, where it used to live.
    """
    train, val, _fell_back = plan_split(image_names, val_pct, groups)
    return train, val
