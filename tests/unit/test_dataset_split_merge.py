"""Folding near-duplicate clusters into the structural grouping (#82, ADR-045).

ADR-044 left one case open and said so: a folder of video frames extracted as
ordinary files (``clip_F00001.png``) does not group, because the dot means
"independent file" and closing that by name would throw legitimately separate
photographs called ``sample_T1.png`` into one bucket.

Embedding clusters close it from the other side -- by what the pixels say
rather than by what the name says -- and this is the machinery that folds them
in. It is also the answer to "what is the curation output *for*": not a report,
a split.
"""

from src.digitalsreeni_image_annotator.core.dataset_split import (
    merge_groups,
    plan_split,
    translate_clusters,
)


# --- merge_groups ----------------------------------------------------------


def test_a_cluster_puts_its_members_in_one_group():
    groups = {"a.png": "a.png", "b.png": "b.png", "c.png": "c.png"}
    merged = merge_groups(groups, [["a.png", "b.png"]])
    assert merged["a.png"] == merged["b.png"]
    assert merged["c.png"] != merged["a.png"]


def test_a_cluster_spanning_two_groups_merges_both_entirely():
    """If one frame of stack A is a near-duplicate of one frame of stack B,
    the two stacks were never safe to split either. Merging only the two named
    frames would leave their siblings free to straddle the split, which is the
    leak this exists to prevent.
    """
    groups = {
        "a_T1": "a", "a_T2": "a",
        "b_T1": "b", "b_T2": "b",
    }
    merged = merge_groups(groups, [["a_T1", "b_T1"]])
    assert len(set(merged.values())) == 1


def test_clusters_chain_transitively():
    groups = {name: name for name in ("a", "b", "c", "d")}
    merged = merge_groups(groups, [["a", "b"], ["b", "c"]])
    assert merged["a"] == merged["b"] == merged["c"]
    assert merged["d"] != merged["a"]


def test_names_outside_the_split_are_ignored():
    """Clusters are computed over the whole project; the grouping describes the
    annotated subset about to be split. Mentioning an image that is not in it
    must not add one."""
    groups = {"a": "a", "b": "b"}
    merged = merge_groups(groups, [["a", "unannotated"]])
    assert set(merged) == {"a", "b"}
    assert merged["a"] != merged["b"]


def test_a_cluster_of_one_changes_nothing():
    groups = {"a": "a", "b": "b"}
    assert merge_groups(groups, [["a"]]) == groups
    assert merge_groups(groups, []) == groups


def test_the_original_grouping_is_not_mutated():
    groups = {"a": "a", "b": "b"}
    merge_groups(groups, [["a", "b"]])
    assert groups == {"a": "a", "b": "b"}


def test_a_merged_group_never_straddles_the_split():
    """The property the whole feature exists for, end to end."""
    names = [f"shot{index:02d}.png" for index in range(20)]
    groups = merge_groups(
        {name: name for name in names}, [names[:8], names[8:12]]
    )
    _train, val, _fell_back = plan_split(names, 30, groups)

    for cluster in (names[:8], names[8:12]):
        held = [name for name in cluster if name in val]
        assert len(held) in (0, len(cluster)), cluster


# --- translate_clusters ----------------------------------------------------


def test_clusters_are_rewritten_as_split_keys():
    """The SAM path splits on "{index}:{name}" keys. Handing it name-keyed
    clusters would match nothing at all -- and silently, since merge_groups
    ignores names it does not recognise."""
    names_by_key = {"0:a": "a", "1:b": "b", "2:c": "c"}
    assert translate_clusters([["a", "b"]], names_by_key) == [["0:a", "1:b"]]


def test_every_entry_sharing_a_name_is_translated():
    """Two SampleGroups can share a name (a prepared folder holding `a.png` and
    `a.jpg`). Both entries carry the cluster."""
    names_by_key = {"0:a": "a", "1:a": "a", "2:b": "b"}
    assert translate_clusters([["a", "b"]], names_by_key) == [
        ["0:a", "1:a", "2:b"]
    ]


def test_a_cluster_that_translates_to_nothing_is_dropped():
    names_by_key = {"0:a": "a"}
    assert translate_clusters([["x", "y"]], names_by_key) == []
    # One key left is not a cluster: it constrains nothing.
    assert translate_clusters([["a", "y"]], names_by_key) == []
