"""The vectorised near-duplicate pass is equivalent to the loop it replaced (#82).

``cluster`` used to be a pure-Python double loop over every pair, recomputing
both vector norms each time. It is now a blocked NumPy pass with an incremental
component labelling, and ``outliers`` and ``modes`` come off the same pass.

That is a large enough rewrite that "the old fixtures still pass" is not
evidence of anything -- those fixtures have three images in them. This file is
the actual net: a naive reference implementation, checked against the fast one
at *every threshold where the answer can differ*.
"""

import random

import pytest

from src.digitalsreeni_image_annotator.core import similarity


def _reference_clusters(embeddings, threshold):
    """The implementation this module used to have: every pair, union-find.

    Deliberately naive, and kept here rather than in the module. Its only job
    is to be obviously correct.
    """
    names = sorted(embeddings)
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            name = parent[name]
        return name

    for index, first in enumerate(names):
        for second in names[index + 1:]:
            pair = similarity.cosine_similarity(
                embeddings[first], embeddings[second]
            )
            if pair >= threshold:
                root_first, root_second = find(first), find(second)
                if root_first != root_second:
                    parent[root_second] = root_first

    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    clusters = [sorted(group) for group in groups.values() if len(group) > 1]
    clusters.sort(key=lambda group: (-len(group), group[0]))
    return clusters


def _random_embeddings(count, dimension=6, seed=0):
    rng = random.Random(seed)
    return {
        f"img{index:03d}": similarity.l2_normalise(
            [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        )
        for index in range(count)
    }


def _pairwise(embeddings):
    names = sorted(embeddings)
    return [
        similarity.cosine_similarity(embeddings[first], embeddings[second])
        for index, first in enumerate(names)
        for second in names[index + 1:]
    ]


def _interesting_thresholds(embeddings):
    """Every threshold at which the clustering can change.

    Midpoints between consecutive distinct pairwise similarities, plus one
    outside each end. Sweeping these covers every distinct outcome the
    function has, which is a stronger check than random thresholds -- and it
    keeps the comparison away from the exact boundaries, where float32 and
    float64 arithmetic legitimately disagree in the last bit.
    """
    values = sorted(set(_pairwise(embeddings)))
    midpoints = [
        (low + high) / 2 for low, high in zip(values, values[1:])
    ]
    return [values[0] - 0.1] + midpoints + [values[-1] + 0.1]


# --- equivalence -----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_clustering_matches_the_reference_at_every_distinct_threshold(seed):
    embeddings = _random_embeddings(14, seed=seed)
    thresholds = _interesting_thresholds(embeddings)
    # Sanity: the sweep must actually exercise a range of outcomes, or this
    # test could pass by comparing "no clusters" against "no clusters".
    sizes = {
        len(similarity.cluster(embeddings, value)) for value in thresholds
    }
    assert len(sizes) > 1

    for value in thresholds:
        assert similarity.cluster(embeddings, value) == _reference_clusters(
            embeddings, value
        ), value


def test_outliers_match_a_naive_nearest_neighbour_scan():
    embeddings = _random_embeddings(20, seed=7)
    names = sorted(embeddings)
    for value in _interesting_thresholds(embeddings):
        expected = [
            name
            for name in names
            if max(
                similarity.cosine_similarity(embeddings[name], embeddings[other])
                for other in names
                if other != name
            )
            < value
        ]
        assert similarity.outliers(embeddings, value) == expected, value


def test_a_chain_of_near_duplicates_stays_one_cluster():
    """Transitivity at a scale the three-image fixture cannot show.

    A slow pan: each frame is close to its neighbour and far from the ends.
    This is the case connected components exists for, and the case an
    edge-list implementation would have paid O(n^2) edges to discover.
    """
    embeddings = {
        f"frame{index:03d}": similarity.l2_normalise([1.0, index * 0.02])
        for index in range(60)
    }
    ends = similarity.cosine_similarity(
        embeddings["frame000"], embeddings["frame059"]
    )
    assert ends < 0.8

    clusters = similarity.cluster(embeddings, threshold=0.99)
    assert len(clusters) == 1
    assert len(clusters[0]) == 60


def test_the_threshold_is_inclusive():
    """A pair exactly at the threshold clusters. The slider's readout is the
    number the user is reasoning about, so "0.90" must mean "0.90 counts".
    """
    embeddings = {"a": [1.0, 0.0], "b": [0.6, 0.8]}
    exact = similarity.cosine_similarity(embeddings["a"], embeddings["b"])
    assert exact == pytest.approx(0.6)

    assert similarity.cluster(embeddings, threshold=0.6) == [["a", "b"]]
    assert similarity.outliers(embeddings, threshold=0.6) == []


def test_raw_model_output_is_normalised_before_comparing():
    """`cosine_similarity` normalised defensively and callers relied on it.

    The fixture is chosen so the dot product and the cosine **disagree about
    which pair is closest**, not merely about magnitudes:

    * `axis` and `short` are parallel -- cosine 1.0 -- but their dot product is
      0.15, below any usable threshold;
    * `axis` and `diagonal` are 45 degrees apart -- cosine 0.71 -- but their dot
      product is 9.

    So without normalisation the answer is not merely inflated, it is the
    opposite one. An earlier version of this test used axis-aligned vectors,
    where dropping the normalisation scaled every similarity without changing
    which pairs cleared the threshold -- and the mutation survived.
    """
    embeddings = {
        "axis": [3.0, 0.0],
        "short": [0.05, 0.0],
        "diagonal": [3.0, 3.0],
    }
    assert similarity.cluster(embeddings, threshold=0.99) == [["axis", "short"]]
    assert similarity.outliers(embeddings, threshold=0.99) == ["diagonal"]


def test_the_matrix_is_built_without_a_second_copy():
    """The shared entry point, which sets the real ceiling for everything else.

    `(matrix * matrix).sum(axis=1)` materialises a full second copy purely to
    sum it away, and every blocked pairwise routine calls this before its
    blocking starts — so it doubled the peak of all of them and made three
    functions with very different pairwise costs measure identically.

    A wide embedding is the point here: the other memory tests use d=4 so the
    matrix is negligible and the *pairwise* term is what they measure, which is
    exactly why neither of them could see this.
    """
    import tracemalloc

    import numpy

    count, dimension = 4000, 768
    rng = numpy.random.default_rng(2)
    names = [f"n{index:05d}" for index in range(count)]
    embeddings = {
        name: rng.random(dimension).astype(numpy.float32) for name in names
    }
    matrix_bytes = count * dimension * 4

    tracemalloc.start()
    try:
        matrix = similarity._stack(names, embeddings)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert matrix.shape == (count, dimension)
    assert peak < matrix_bytes * 1.5, f"{peak / 1e6:.0f} MB for a {matrix_bytes / 1e6:.0f} MB matrix"


def test_the_main_pass_is_blocked():
    """The routine the module header quotes its headline figure for.

    `representative` got a memory test after a review found it unblocked;
    `_scan` -- which every one of `cluster`, `outliers`, `modes` and `analyse`
    runs -- did not, and mutating its block height away survived the whole
    suite. Unblocked at the supported ceiling it allocates 1.6 GB for the
    similarities plus 400 MB per boolean mask.

    The dimension is deliberately tiny so the (n, d) matrix is negligible and
    what is measured is the pairwise term.
    """
    import tracemalloc

    import numpy

    count = 4000
    assert similarity._block_rows(count) < count, "pick a size that blocks"
    rng = numpy.random.default_rng(1)
    embeddings = {
        f"n{index:05d}": rng.random(4).astype(numpy.float32)
        for index in range(count)
    }

    tracemalloc.start()
    try:
        result = similarity.analyse(embeddings, 0.99, 0.8)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert result["modes"]
    # The unblocked similarity matrix alone is count^2 * 4 = 64 MB, before the
    # two boolean masks.
    assert peak < count * count * 4 * 0.75, f"{peak / 1e6:.0f} MB"


def test_the_representative_pass_is_blocked(monkeypatch):
    """Peak memory, asserted rather than claimed.

    A single video clusters into ONE component, so `cluster_names` is routinely
    the whole dataset and "just multiply the cluster" is a k x k allocation --
    1.6 GB at the supported ceiling, once per cluster, on the GUI thread. Every
    other pairwise routine here was blocked; this one was not, and no test
    noticed because blocking changes memory and nothing else.
    """
    import tracemalloc

    import numpy

    count = 4000
    assert similarity._block_rows(count) < count, "pick a size that blocks"
    names = [f"n{index:05d}" for index in range(count)]
    rng = numpy.random.default_rng(0)
    embeddings = {
        name: rng.random(4).astype(numpy.float32) for name in names
    }

    tracemalloc.start()
    try:
        chosen = similarity.representative(names, embeddings)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert chosen in embeddings
    # The full product alone is count^2 * 4 bytes = 64 MB; blocked it is ~32.
    assert peak < count * count * 4 * 0.75, f"{peak / 1e6:.0f} MB"


# --- blocking --------------------------------------------------------------


def test_block_size_does_not_change_the_answer(monkeypatch):
    """The row-block height is a memory knob, never a semantic one.

    Forcing single-row blocks is what a project of hundreds of thousands of
    images would do to the block size, so this is the shape of the code that
    runs at the top of the supported range -- not an artificial case.
    """
    embeddings = _random_embeddings(25, seed=11)
    threshold = 0.5
    expected = similarity.cluster(embeddings, threshold)
    expected_outliers = similarity.outliers(embeddings, threshold)
    assert expected, "pick a threshold that actually clusters something"

    monkeypatch.setattr(similarity, "_MAX_BLOCK_ELEMENTS", 1)
    assert similarity._block_rows(25) == 1
    assert similarity.cluster(embeddings, threshold) == expected
    assert similarity.outliers(embeddings, threshold) == expected_outliers


def test_the_block_height_is_never_zero():
    """A wide embedding must not divide the block height to nothing -- a zero
    step would loop forever rather than fail."""
    assert similarity._block_rows(10**9) >= 1
    assert similarity._block_rows(0) >= 1


# --- degenerate vectors ----------------------------------------------------


def test_a_vector_of_the_wrong_length_resembles_nothing():
    """Mixed dimensions mean mixed models, which the model-keyed cache
    prevents. If one arrives anyway -- a hand-edited cache -- it is reported as
    resembling nothing, exactly as the old length check did, rather than
    crashing a run halfway through."""
    embeddings = {
        "a": similarity.l2_normalise([1.0, 0.0]),
        "b": similarity.l2_normalise([1.0, 0.0]),
        "odd": [1.0, 0.0, 0.0],
    }
    assert similarity.cluster(embeddings, threshold=0.9) == [["a", "b"]]
    assert similarity.outliers(embeddings, threshold=0.9) == ["odd"]


def test_a_blank_image_resembles_nothing_including_another_blank():
    """A zero vector has no direction, so it is not similar to anything -- and
    two of them are not near-duplicates of each other either. That was the old
    behaviour (a zero norm returned 0.0) and it stays."""
    embeddings = {"blank1": [0.0, 0.0], "blank2": [0.0, 0.0]}
    assert similarity.cluster(embeddings, threshold=0.5) == []
    assert similarity.outliers(embeddings, threshold=0.5) == [
        "blank1",
        "blank2",
    ]


def test_numpy_vectors_are_accepted():
    """The controller holds float32 arrays rather than Python float lists -- at
    the supported ceiling that is 60 MB instead of roughly 500 MB. `_stack`
    must therefore never test a vector for truthiness."""
    numpy = pytest.importorskip("numpy")
    embeddings = {
        "a": numpy.array([1.0, 0.0], dtype=numpy.float32),
        "b": numpy.array([1.0, 0.0], dtype=numpy.float32),
        "c": numpy.array([0.0, 1.0], dtype=numpy.float32),
    }
    assert similarity.cluster(embeddings, threshold=0.95) == [["a", "b"]]
    assert similarity.representative(["a", "b"], embeddings) == "a"
