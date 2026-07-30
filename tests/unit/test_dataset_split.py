"""Group-aware train/val splitting (issue #81, ADR-044).

The bug these tests exist for is invisible at runtime. A name-keyed split over
a video's frames produces a perfectly ordinary-looking dataset, trains without
a warning, and reports validation metrics that are simply too good — the more
redundant the data, the better they look. Nothing fails; the numbers just stop
meaning what everyone reads them as.

So the load-bearing test here is :func:`test_no_group_ever_straddles_the_split`.
The rest establish that the grouping is derived correctly in the first place.
"""

import subprocess
import sys

from src.digitalsreeni_image_annotator.core import dataset_split
from src.digitalsreeni_image_annotator.core.dataset_split import (
    assign_train_val,
    derive_groups,
    plan_split,
)


class _FakeSliceList:
    """Stand-in for ``LazySliceList``: only ``.names`` is ever touched."""

    def __init__(self, names):
        self.names = list(names)


def _frames(base, count):
    return [f"{base}_F{i:05d}" for i in range(count)]


def _buckets(names, groups):
    """``{group_key: {name, ...}}`` for asserting on whole groups."""
    buckets = {}
    for name in names:
        buckets.setdefault(groups[name], set()).add(name)
    return buckets


# --- Qt-free guarantee -----------------------------------------------------


def test_the_split_imports_without_qt():
    """``io.export_formats`` imports this module, and the headless CLI imports
    that — a stray Qt import here would make a CI export need a display.

    Specifically it must not reach ``core.slice_cache``, which pulls in
    ``core.image_utils`` and therefore ``QImage``.
    """
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.dataset_split as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- deriving the grouping -------------------------------------------------


def test_slices_group_by_their_stack():
    """``image_slices`` is keyed by the ext-stripped base name, so the mapping
    is exact — no parsing, no pixel work."""
    image_slices = {"stack": _FakeSliceList(["stack_T1_Z1", "stack_T1_Z2"])}
    groups = derive_groups(["stack_T1_Z1", "stack_T1_Z2", "photo.png"], image_slices)
    assert groups["stack_T1_Z1"] == "stack"
    assert groups["stack_T1_Z2"] == "stack"
    assert groups["photo.png"] == "photo.png"


def test_a_plain_list_slice_collection_also_groups():
    """Legacy call sites and several tests still hand in ``[(name, qimage)]``."""
    image_slices = {"stack": [("stack_Z1", None), ("stack_Z2", None)]}
    groups = derive_groups(["stack_Z1", "stack_Z2"], image_slices)
    assert groups["stack_Z1"] == groups["stack_Z2"] == "stack"


def test_slice_names_group_without_any_image_slices():
    """The fallback that protects the CLI, which passes an empty mapping, and
    an .iap whose stack was never materialised this session."""
    names = ["stack_T1_Z1", "stack_T2_Z1", "other_T1_Z1"]
    groups = derive_groups(names)
    assert groups["stack_T1_Z1"] == groups["stack_T2_Z1"] == "stack"
    assert groups["other_T1_Z1"] == "other"


def test_video_frames_group_by_recording():
    names = _frames("clip", 3) + _frames("other", 2)
    groups = derive_groups(names)
    assert {groups[n] for n in _frames("clip", 3)} == {"clip"}
    assert {groups[n] for n in _frames("other", 2)} == {"other"}


def test_a_regular_image_is_its_own_group():
    """Regular names keep their extension; the dot is what tells them apart
    from a slice name, the same signal the exporters already use."""
    names = ["a_T1.png", "b.jpg", "c.png"]
    groups = derive_groups(names)
    assert groups == {"a_T1.png": "a_T1.png", "b.jpg": "b.jpg", "c.png": "c.png"}


def test_an_exact_mapping_beats_the_name_heuristic():
    """A stack whose base name itself looks like a slice suffix is grouped by
    what ``image_slices`` says, not by what the regex guesses."""
    image_slices = {"run_T1": _FakeSliceList(["run_T1_Z1", "run_T1_Z2"])}
    groups = derive_groups(["run_T1_Z1", "run_T1_Z2"], image_slices)
    assert groups["run_T1_Z1"] == groups["run_T1_Z2"] == "run_T1"


# --- the split itself ------------------------------------------------------


def test_no_group_ever_straddles_the_split():
    """THE test. Two recordings plus loose photos: every frame of a recording
    has to land on one side, whichever side that is."""
    names = _frames("clipA", 30) + _frames("clipB", 30) + [
        f"photo{i}.png" for i in range(10)
    ]
    groups = derive_groups(names)
    train, val, fell_back = plan_split(names, 20, groups)

    assert not fell_back
    for members in _buckets(names, groups).values():
        assert members <= train or members <= val, members


def test_a_video_project_holds_out_a_whole_recording():
    names = _frames("clipA", 20) + _frames("clipB", 20)
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 50, groups)
    assert train and val
    assert train.isdisjoint(val)
    assert train | val == set(names)
    # Literal, not recomputed: the property test measures locality against a
    # target it derives the same way `plan_split` does, so it cannot catch a
    # wrong target. One hard number closes that loop.
    assert len(val) == 20


def test_a_single_recording_falls_back_and_says_so():
    """One video is the case where no honest split exists. Returning an empty
    val set would be truthful but makes the trainer silently drop validation
    and early stopping (ADR-028), so the flag carries the news instead."""
    names = _frames("clip", 20)
    groups = derive_groups(names)
    train, val, fell_back = plan_split(names, 20, groups)
    assert fell_back
    assert len(val) == 4 and len(train) == 16


def test_neither_side_is_ever_empty_even_with_lopsided_groups():
    """A group is indivisible, so a single huge one could otherwise swallow
    the whole dataset and leave train empty — which is not a split at all."""
    names = _frames("big", 18) + ["a.png", "b.png"]
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 80, groups)
    assert train and val
    assert train | val == set(names)


def test_a_very_high_percentage_still_leaves_something_to_train_on():
    """The bound that stops the hill-climb draining train had no test at all:
    relaxing it from `<` to `<=` left all 1428 other tests green and produced
    `train=0, val=10`. That is the same empty-training-set failure this change
    already shipped once, so it gets its own case at the percentage that
    reaches it.

    Above 50% is reachable in the app: the export prompt accepts 0-100 and the
    SAM dialog blocks only 0% train.
    """
    names, groups = [], {}
    for key, size in (("g0", 3), ("g1", 7)):
        for member in range(size):
            name = f"{key}_m{member}"
            names.append(name)
            groups[name] = key

    for val_pct in (60, 75, 90, 100):
        train, val, _ = plan_split(names, val_pct, groups)
        assert train, f"train emptied at {val_pct}%"
        assert val, f"val emptied at {val_pct}%"
        assert train | val == set(names)


def test_the_group_split_is_deterministic():
    names = _frames("clipA", 10) + _frames("clipB", 10) + ["x.png", "y.png"]
    groups = derive_groups(names)
    first = plan_split(names, 30, groups)
    second = plan_split(list(reversed(names)), 30, groups)
    assert first[:2] == second[:2]


def test_grouping_by_identity_matches_the_ungrouped_split():
    """The compatibility guarantee: every name in its own group is exactly the
    historical per-name split, so `groups=None` callers are unaffected."""
    names = [f"img_{i:03d}.png" for i in range(37)]
    assert assign_train_val(names, 30) == assign_train_val(
        names, 30, {name: name for name in names}
    )


def test_zero_split_never_reports_a_fallback():
    """With no val set requested there is nothing to warn about."""
    names = _frames("clip", 10)
    assert plan_split(names, 0, derive_groups(names)) == (set(names), set(), False)


def test_the_split_covers_and_partitions_every_name():
    names = _frames("clipA", 7) + _frames("clipB", 5) + ["solo.png"]
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 25, groups)
    assert train.isdisjoint(val)
    assert train | val == set(names)


def test_slice_base_leaves_a_dotted_name_alone():
    """Guards the one heuristic in the module against widening.

    ``photo.raw_T1`` is the case that actually pins the dot check: the others
    are refused by the end-anchored regex on their own, so without this line
    deleting the guard entirely left the whole suite green.
    """
    assert dataset_split._slice_base("photo_T1.png") is None
    assert dataset_split._slice_base("photo.raw_T1") is None
    assert dataset_split._slice_base("stack_T1_Z2") == "stack"
    assert dataset_split._slice_base("plain_name") is None


def test_the_heuristic_only_claims_real_dimension_letters():
    """Restricting the suffix to the letters ``DimensionDialog`` assigns
    (T/Z/C/S) plus F for video frames keeps most well-plate names intact.
    Matching any ``[A-Z]\\d+`` put every well of a plate under one key.
    """
    assert dataset_split._slice_base("Plate1_A1_T1_Z1") == "Plate1_A1"
    assert dataset_split._slice_base("Plate1_H12_Z1") == "Plate1_H12"
    assert dataset_split._slice_base("clip_F00042") == "clip"
    assert dataset_split._slice_base("mouse_S1_T1") == "mouse"


def test_the_name_heuristic_is_ambiguous_where_the_name_is_ambiguous():
    """Documents a real limit rather than pretending it away.

    ``Plate1_C1_Z1`` is indistinguishable, from the name alone, from a stack
    ``Plate1`` with a channel and a Z dimension — so wells in rows C, F, S, T
    and Z collapse into the stack key. Nothing in a filename can resolve that.

    It is the *safe* ambiguity: over-grouping costs split granularity, and
    where it collapses far enough the ``fell_back`` flag fires and the user is
    told. Under-grouping would silently reopen the leak instead. The exact
    mapping below is what removes the guesswork.
    """
    names = [f"Plate1_{row}{col}_Z1" for row in "ABCDEFGH" for col in range(1, 13)]
    groups = derive_groups(names)
    assert len(set(groups.values())) == 73  # 96 wells, rows C and F absorbed
    _train, _val, fell_back = plan_split(names, 20, groups)
    assert not fell_back


def test_a_loaded_project_groups_wells_exactly():
    """With ``image_slices`` present — i.e. in the app, as opposed to the CLI —
    the ambiguity above does not arise: each well is its own collection."""
    names = [f"Plate1_{row}{col}_Z1" for row in "ABCDEFGH" for col in range(1, 13)]
    image_slices = {
        name.rsplit("_", 1)[0]: _FakeSliceList([name]) for name in names
    }
    groups = derive_groups(names, image_slices)
    assert len(set(groups.values())) == 96


# --- how close the split lands to what was asked for -----------------------


def _val_count(total, val_pct):
    """The target `plan_split` aims at — same formula, so the property test
    measures the split rather than restating it."""
    return max(1, min(total - 1, round(total * val_pct / 100)))


def test_the_split_size_is_locally_optimal():
    """A group is indivisible, so the requested percentage cannot always be
    hit. What *is* guaranteed: no single group, added or substituted, would
    land closer to the target.

    This is the test that would have caught the original size-blind greedy,
    which delivered 1 %, 9 % and 67 % for a requested 20 % on ordinary
    video-and-photo projects while every cohesion assertion stayed green.
    """
    import random

    rng = random.Random(20260727)
    for trial in range(300):
        sizes = [rng.choice([1, 1, 1, 2, 5, 30, 64, 100]) for _ in range(rng.randint(2, 9))]
        names, groups = [], {}
        for index, size in enumerate(sizes):
            for member in range(size):
                name = f"g{index}_m{member}"
                names.append(name)
                groups[name] = f"g{index}"
        # Past 50 too: the non-empty-side bounds are only approached up there,
        # and the app lets the user go to 100.
        val_pct = rng.choice([10, 20, 25, 30, 50, 60, 75, 90])
        target = _val_count(len(names), val_pct)

        train, val, _ = plan_split(names, val_pct, groups)
        buckets = _buckets(names, groups)
        held = len(val)
        distance = abs(held - target)

        message = f"trial {trial}: sizes={sizes} pct={val_pct} held={held} target={target}"
        inside = [m for m in buckets.values() if m <= val]
        outside = [m for m in buckets.values() if not m <= val]

        for members in outside:
            # Adding one more group must not improve things (unless it would
            # empty train, which the split refuses)...
            if len(val) + len(members) < len(names):
                assert abs(held + len(members) - target) >= distance, message
        for members in inside:
            # ...nor dropping one (unless it would empty val)...
            if len(inside) > 1:
                assert abs(held - len(members) - target) >= distance, message
        for gone in inside:
            for came in outside:
                # ...nor swapping one for another, which is the move the first
                # version of this test forgot and the implementation therefore
                # never made: two 60-frame videos and two 30-frame stacks at
                # 50 % delivered 33 %, with a swap sitting exactly on target.
                swapped = held - len(gone) + len(came)
                assert abs(swapped - target) >= distance, message
        for members in buckets.values():
            # ...nor holding out that group alone.
            assert abs(len(members) - target) >= distance, message


def test_a_swap_is_taken_when_it_lands_on_the_target():
    """The concrete case the property test was blind to."""
    names, groups = [], {}
    for key, size in (("run1", 60), ("run2", 30), ("run3", 60), ("run4", 30)):
        for member in range(size):
            name = f"{key}_m{member}"
            names.append(name)
            groups[name] = key

    _train, val, _ = plan_split(names, 50, groups)
    assert len(val) == 90  # 50% of 180, hit exactly by one 60 plus one 30


def test_every_group_stays_whole_across_randomised_projects():
    import random

    rng = random.Random(4711)
    for _ in range(200):
        sizes = [rng.randint(1, 40) for _ in range(rng.randint(2, 8))]
        names, groups = [], {}
        for index, size in enumerate(sizes):
            for member in range(size):
                name = f"g{index}_m{member}"
                names.append(name)
                groups[name] = f"g{index}"
        train, val, _ = plan_split(names, rng.choice([10, 20, 40, 70, 95]), groups)
        assert train and val
        assert train.isdisjoint(val)
        assert train | val == set(names)
        for members in _buckets(names, groups).values():
            assert members <= train or members <= val


# --- the warning ------------------------------------------------------------
#
# `split_warning` is pure text and lives in core, not on the controller, so the
# CLI can emit the identical wording (ADR-044). These need no QApplication.


def test_no_warning_when_the_grouping_works():
    names = _frames("clipA", 10) + _frames("clipB", 10)
    assert dataset_split.split_warning(names, 20) is None


def test_no_warning_when_no_validation_set_was_asked_for():
    assert dataset_split.split_warning(_frames("clip", 10), 0) is None


def test_a_single_group_warns_that_the_metrics_are_optimistic():
    message = dataset_split.split_warning(_frames("clip", 10), 20)
    assert message is not None
    assert "optimistic" in message


def test_a_training_set_of_one_group_is_reported():
    """Holding out a percentage by image count can route every small group to
    validation when one dominates — optimal by that count, useless as a
    dataset, and silent because the grouping technically worked."""
    names = _frames("clip", 200) + [f"p{i}.png" for i in range(20)]
    message = dataset_split.split_warning(names, 20)
    assert message is not None
    assert "single group" in message


def test_a_healthy_multi_recording_split_stays_quiet():
    names = _frames("a", 30) + _frames("b", 30) + _frames("c", 30) + _frames("d", 30)
    assert dataset_split.split_warning(names, 25) is None


def test_the_warning_is_reachable_without_qt():
    """It is the CLI's copy too, so it must not have followed the dialog into
    a Qt-importing module — which is where it started out."""
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "from digitalsreeni_image_annotator.core.dataset_split import split_warning;"
        "assert split_warning(['c_F00001', 'c_F00002'], 20);"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_preview_counts_only_what_the_export_will_write(tmp_path):
    """The preview and the export must partition the *same* names.

    Computed separately they drifted: a project holding an unopened video's
    frames previewed two groups and stayed quiet, while the export saw one
    group, fell back to the per-name split, and leaked. Both now go through
    ``exportable_annotated_names``.
    """
    from src.digitalsreeni_image_annotator.controllers.io_controller import (
        annotated_image_names,
    )

    photo = tmp_path / "a.png"
    photo.write_bytes(b"")

    class _MainWindow:
        all_annotations = {
            "a.png": {"cell": [{"bbox": [0, 0, 1, 1]}]},
            "b.png": {},  # unannotated
            "ghost_F00001": {"cell": [{"bbox": [0, 0, 1, 1]}]},  # never loaded
        }
        image_paths = {"a.png": str(photo)}
        slices = []
        image_slices = {}

    # The ghost frame has no pixels anywhere, so the export skips it and it
    # must not consume a slot in the split budget either.
    assert annotated_image_names(_MainWindow()) == ["a.png"]
