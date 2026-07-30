"""Embedding-based near-duplicate clustering and caching (issue #72).

The clustering is Qt-free and model-free — it takes plain float vectors — so it
is tested on synthetic ones. That is the point of the split: the backend can be
swapped from CLIP to DINOv2 and compared without a single clustering test
changing.

The cache tests carry more weight than they look like they should. A curation
tool you can only afford to run once is a curation tool nobody uses, so "a
second run over unchanged data is nearly instant" is a feature requirement, not
an optimisation — and content-hash keying is what delivers it.
"""

import math
import subprocess
import sys

import pytest

from src.digitalsreeni_image_annotator.core import similarity
from src.digitalsreeni_image_annotator.inference.embedding_utils import (
    EmbeddingCache,
    content_hash,
)


def _unit(*components):
    return similarity.l2_normalise(list(components))


# --- Qt-free guarantee -----------------------------------------------------


def test_the_clustering_imports_without_qt():
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.similarity as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- vector maths ----------------------------------------------------------


def test_normalisation_produces_a_unit_vector():
    vector = similarity.l2_normalise([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


def test_a_zero_vector_normalises_to_itself():
    """A blank image is a legitimate input, not a division by zero."""
    assert similarity.l2_normalise([0.0, 0.0]) == [0.0, 0.0]


def test_identical_vectors_are_maximally_similar():
    vector = _unit(1.0, 2.0, 3.0)
    assert similarity.cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_are_not_similar():
    assert similarity.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_similarity_of_mismatched_or_empty_vectors_is_zero():
    assert similarity.cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert similarity.cosine_similarity([], [1.0]) == 0.0
    assert similarity.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- clustering ------------------------------------------------------------


def test_identical_images_cluster():
    embeddings = {"a.png": _unit(1, 0), "b.png": _unit(1, 0), "c.png": _unit(0, 1)}
    clusters = similarity.cluster(embeddings, threshold=0.95)
    assert clusters == [["a.png", "b.png"]]


def test_singletons_are_omitted():
    """An image similar to nothing is not a near-duplicate cluster; including
    every one would bury the handful of findings that matter."""
    embeddings = {"a.png": _unit(1, 0), "b.png": _unit(0, 1)}
    assert similarity.cluster(embeddings, threshold=0.95) == []


def test_clustering_is_transitive():
    """A~B and B~C groups all three even if A and C fall below the threshold —
    the right behaviour for a burst of frames drifting slowly."""
    embeddings = {
        "a.png": _unit(1.0, 0.0),
        "b.png": _unit(1.0, 0.30),
        "c.png": _unit(1.0, 0.62),
    }
    assert similarity.cosine_similarity(
        embeddings["a.png"], embeddings["c.png"]
    ) < 0.9
    assert similarity.cluster(embeddings, threshold=0.9) == [
        ["a.png", "b.png", "c.png"]
    ]


def test_clusters_are_ordered_largest_first():
    embeddings = {
        "a1": _unit(1, 0), "a2": _unit(1, 0), "a3": _unit(1, 0),
        "b1": _unit(0, 1), "b2": _unit(0, 1),
    }
    clusters = similarity.cluster(embeddings, threshold=0.95)
    assert [len(group) for group in clusters] == [3, 2]


def test_raising_the_threshold_splits_a_cluster():
    """Re-clustering is arithmetic over vectors already in memory, which is
    what makes the slider instant."""
    embeddings = {"a": _unit(1.0, 0.0), "b": _unit(1.0, 0.35)}
    assert similarity.cluster(embeddings, threshold=0.90) == [["a", "b"]]
    assert similarity.cluster(embeddings, threshold=0.99) == []


def test_clustering_an_empty_or_single_set_is_safe():
    assert similarity.cluster({}) == []
    assert similarity.cluster(None) == []
    assert similarity.cluster({"only": _unit(1, 0)}) == []


# --- representative and outliers -------------------------------------------


def test_the_representative_is_the_most_central_member():
    """The medoid, not the first alphabetically — the frame that best stands in
    for the group."""
    embeddings = {
        "left": _unit(1.0, 0.0),
        "middle": _unit(1.0, 0.2),
        "right": _unit(1.0, 0.4),
    }
    assert similarity.representative(
        ["left", "middle", "right"], embeddings
    ) == "middle"


def test_representative_of_a_trivial_cluster():
    assert similarity.representative(["a"], {"a": _unit(1, 0)}) == "a"
    assert similarity.representative([], {}) is None


def test_outliers_are_the_images_nothing_resembles():
    embeddings = {
        "a": _unit(1, 0), "b": _unit(1, 0), "lonely": _unit(0, 1),
    }
    assert similarity.outliers(embeddings, threshold=0.95) == ["lonely"]


def test_outliers_needs_at_least_two_images():
    assert similarity.outliers({"a": _unit(1, 0)}) == []


# --- summary ---------------------------------------------------------------


def test_summary_counts_what_could_be_skipped():
    """`redundant` is the number that turns a finding into a decision."""
    stats = similarity.summarise([["a", "b", "c"], ["d", "e"]], total_images=10)
    assert stats["clusters"] == 2
    assert stats["clustered_images"] == 5
    assert stats["redundant_images"] == 3
    assert stats["largest_cluster"] == 3
    assert stats["total_images"] == 10


def test_summary_of_no_clusters():
    stats = similarity.summarise([], total_images=4)
    assert stats["redundant_images"] == 0
    assert stats["largest_cluster"] == 0


# --- cache -----------------------------------------------------------------


def test_content_hash_is_stable_for_identical_bytes(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"identical")
    second.write_bytes(b"identical")
    assert content_hash(str(first)) == content_hash(str(second))


def test_content_hash_changes_when_the_bytes_change(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"before")
    before = content_hash(str(path))
    path.write_bytes(b"after")
    assert content_hash(str(path)) != before


def test_content_hash_of_a_missing_file_is_none(tmp_path):
    assert content_hash(str(tmp_path / "nope.bin")) is None


def test_cache_returns_what_it_stored(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    cache.put("CLIP", "hash1", [0.1, 0.2])
    assert cache.get("CLIP", "hash1") == [0.1, 0.2]


def test_cache_misses_on_a_different_content_hash(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    cache.put("CLIP", "hash1", [0.1, 0.2])
    assert cache.get("CLIP", "hash2") is None


def test_cache_misses_across_models(tmp_path):
    """Reusing CLIP vectors for a DINOv2 run would produce clusters that look
    plausible and mean nothing."""
    cache = EmbeddingCache(str(tmp_path))
    cache.put("CLIP", "hash1", [0.1, 0.2])
    assert cache.get("DINOv2", "hash1") is None


def test_cache_survives_a_round_trip_to_disk(tmp_path):
    """This is the feature: a second run over unchanged data recomputes
    nothing."""
    cache = EmbeddingCache(str(tmp_path))
    cache.put("CLIP", "hash1", [0.1, 0.2])
    cache.save()

    reopened = EmbeddingCache(str(tmp_path))
    assert reopened.get("CLIP", "hash1") == [0.1, 0.2]


def test_a_corrupt_cache_is_discarded_not_fatal(tmp_path):
    """A bad cache is a performance problem, never a correctness one."""
    from src.digitalsreeni_image_annotator.inference.embedding_utils import (
        CACHE_FILENAME,
    )

    (tmp_path / CACHE_FILENAME).write_text("{not json at all", encoding="utf-8")
    cache = EmbeddingCache(str(tmp_path))
    assert len(cache) == 0
    cache.put("CLIP", "hash1", [0.5])
    assert cache.get("CLIP", "hash1") == [0.5]


def test_a_cache_without_a_directory_stays_in_memory():
    """A project-less session still benefits within the run, without writing
    litter next to an unrelated working directory."""
    cache = EmbeddingCache(None)
    cache.put("CLIP", "hash1", [0.1])
    assert cache.get("CLIP", "hash1") == [0.1]
    assert cache.path is None
    cache.save()  # must not raise


def test_a_null_hash_is_never_cached(tmp_path):
    """An unreadable file must not poison the cache with a None key."""
    cache = EmbeddingCache(str(tmp_path))
    cache.put("CLIP", None, [0.1])
    assert len(cache) == 0
    assert cache.get("CLIP", None) is None
