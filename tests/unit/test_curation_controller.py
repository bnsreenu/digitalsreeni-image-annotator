"""Curation controller: slice caching, backend switching, and the #71 handoff.

The cache tests are the ones that matter. Before #82 only plain files on disk
were cached, so a project made of video frames -- the case the feature exists
for -- re-embedded every frame on every run. "A curation tool you can only
afford to run once is a curation tool nobody uses" was already in the module
docstring; it just was not true for slices.
"""

import numpy as np
import pytest
from PyQt6.QtWidgets import QListWidget, QWidget

from src.digitalsreeni_image_annotator.controllers.curation_controller import (
    CurationController,
)
from src.digitalsreeni_image_annotator.controllers.review_controller import (
    MODE_DISAGREEMENT,
    MODE_UNCERTAINTY,
)
from src.digitalsreeni_image_annotator.inference.embedding_utils import (
    EmbeddingCache,
    slice_digest,
)


class _Provider:
    def __init__(self, dimensions):
        self.dimensions = list(dimensions)


class _Slices:
    """Duck-types LazySliceList closely enough for collection and embedding."""

    def __init__(self, names, dimensions=None):
        self.names = list(names)
        self.materialised = []
        if dimensions is not None:
            self.provider = _Provider(dimensions)

    def get(self, name):
        self.materialised.append(name)
        return object()  # stands in for a QImage; the embedder is a stub


class _Embedder:
    """Counts calls, so "was this actually recomputed" is answerable."""

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = list(vector)
        self.calls = 0

    def embed_qimage(self, _qimage):
        self.calls += 1
        return list(self.vector)

    def load(self, _model_name):
        pass

    def unload(self):
        pass


class _Review:
    def __init__(self, scores=None):
        self.scores = scores or {}

    def has_scores(self):
        return bool(self.scores)

    def score_for(self, name):
        record = self.scores.get(name)
        return record["score"] if record else None

    def mode_for(self, name):
        record = self.scores.get(name)
        return record["mode"] if record else None


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.all_images = []
        self.image_paths = {}
        self.image_slices = {}
        self.image_list = QListWidget()
        self.current_project_file = None


@pytest.fixture
def controller(qtbot):
    window = _Window()
    qtbot.addWidget(window)
    made = CurationController(window)
    made.embedder = _Embedder()
    return made


def _add_stack(window, file_name, path, slice_names, dimensions=None):
    base = file_name.rsplit(".", 1)[0]
    if not any(info["file_name"] == file_name for info in window.all_images):
        window.all_images.append({"file_name": file_name})
    window.image_paths[file_name] = path
    window.image_slices[base] = _Slices(slice_names, dimensions)
    return window.image_slices[base]


# --- collection ------------------------------------------------------------


def test_slices_carry_their_source_path(controller, tmp_path):
    """Without the source path a slice has nothing to key a cache entry on."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pretend video")
    _add_stack(controller.mw, "clip.mp4", str(video), ["clip_F00000", "clip_F00001"])

    items = controller.collect_work_items()

    assert [name for name, _kind, _payload in items] == [
        "clip_F00000",
        "clip_F00001",
    ]
    assert all(payload[2] == str(video) for _n, _k, payload in items)


def test_a_stack_without_a_path_still_embeds(controller):
    """No path means no cache key, which must not mean no analysis."""
    controller.mw.all_images.append({"file_name": "stack.tif"})
    controller.mw.image_slices["stack"] = _Slices(["stack_T1", "stack_T2"])

    items = controller.collect_work_items()
    assert len(items) == 2
    cache = EmbeddingCache(None)
    vector, from_cache = controller._embed_one("slice", items[0][2], cache)
    assert vector is not None and not from_cache
    assert len(cache) == 0


# --- slice caching (#82) ---------------------------------------------------


def test_a_slice_is_cached_and_the_second_run_recomputes_nothing(
    controller, tmp_path
):
    """THE point of the change: video frames used to be re-embedded every
    single run, and they are the primary case."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pretend video")
    _add_stack(controller.mw, "clip.mp4", str(video), ["clip_F00000", "clip_F00001"])
    cache = EmbeddingCache(str(tmp_path))

    items = controller.collect_work_items()
    for _name, kind, payload in items:
        controller._embed_one(kind, payload, cache)
    assert controller.embedder.calls == 2
    assert len(cache) == 2

    # Second run: fresh digest memo, same files.
    controller._digests = {}
    for _name, kind, payload in items:
        _vector, from_cache = controller._embed_one(kind, payload, cache)
        assert from_cache
    assert controller.embedder.calls == 2, "a cached slice was re-embedded"


def test_a_cached_slice_is_never_materialised(controller, tmp_path):
    """The cache has to spare the *decode* too, not just the forward pass.
    Decoding a frame is the expensive half for a long video."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pretend video")
    slices = _add_stack(
        controller.mw, "clip.mp4", str(video), ["clip_F00000"]
    )
    cache = EmbeddingCache(str(tmp_path))
    payload = controller.collect_work_items()[0][2]

    controller._embed_one("slice", payload, cache)
    assert slices.materialised == ["clip_F00000"]

    controller._embed_one("slice", payload, cache)
    assert slices.materialised == ["clip_F00000"], "a cached frame was decoded"


def test_the_source_file_is_hashed_once_not_once_per_frame(
    controller, tmp_path, monkeypatch
):
    """Hashing a 2 GB video 200 times would cost more than the embeddings it
    saves."""
    import src.digitalsreeni_image_annotator.controllers.curation_controller as module

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pretend video")
    _add_stack(
        controller.mw,
        "clip.mp4",
        str(video),
        [f"clip_F{index:05d}" for index in range(5)],
    )
    hashed = []
    monkeypatch.setattr(
        module, "content_hash", lambda path: hashed.append(path) or "digest"
    )
    cache = EmbeddingCache(None)

    for _name, kind, payload in controller.collect_work_items():
        controller._embed_one(kind, payload, cache)

    assert hashed == [str(video)]


def test_slice_keys_are_distinct_per_frame_and_per_source():
    """Two frames of one video, and the same frame index of two videos, must
    not collide -- either one would serve up another image's vector."""
    assert slice_digest("abc", "clip_F00000") != slice_digest("abc", "clip_F00001")
    assert slice_digest("abc", "clip_F00000") != slice_digest("def", "clip_F00000")
    assert slice_digest(None, "clip_F00000") is None


def test_the_axis_assignment_is_part_of_the_slice_key():
    """A (10, 512, 512) array assigned ZHW and the same array assigned HWZ both
    yield the names `base_Z1`..`base_Z10` -- `_build_index` only emits the
    non-spatial letters -- while indexing a different axis. Same file, same
    name, different pixels; and the cache persists across sessions.
    """
    assert slice_digest("abc", "stack_Z1", ["Z", "H", "W"]) != slice_digest(
        "abc", "stack_Z1", ["H", "W", "Z"]
    )
    # A video provider has no dimensions and needs none.
    assert slice_digest("abc", "clip_F00000", None) == "abc:clip_F00000"


def test_the_controller_passes_the_axis_assignment_into_the_key(
    controller, tmp_path
):
    """`slice_digest` accepting the assignment is worth nothing if the caller
    never reads it off the provider. Same file, same slice name, two different
    assignments: the second must be a cache miss."""
    stack = tmp_path / "stack.tif"
    stack.write_bytes(b"pretend stack")
    cache = EmbeddingCache(str(tmp_path))

    _add_stack(controller.mw, "stack.tif", str(stack), ["stack_Z1"], ["Z", "H", "W"])
    payload = controller.collect_work_items()[0][2]
    controller._embed_one("slice", payload, cache)

    _add_stack(controller.mw, "stack.tif", str(stack), ["stack_Z1"], ["H", "W", "Z"])
    payload = controller.collect_work_items()[0][2]
    _vector, from_cache = controller._embed_one("slice", payload, cache)

    assert not from_cache, "a different axis assignment reused the old vectors"


def test_slice_cache_entries_do_not_cross_models(controller, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"pretend video")
    _add_stack(controller.mw, "clip.mp4", str(video), ["clip_F00000"])
    cache = EmbeddingCache(str(tmp_path))
    payload = controller.collect_work_items()[0][2]

    controller._embed_one("slice", payload, cache)
    controller.set_model("DINOv2 (base)")
    _vector, from_cache = controller._embed_one("slice", payload, cache)

    assert not from_cache, "a DINOv2 run reused CLIP vectors"


# --- backend switching -----------------------------------------------------


def test_switching_model_drops_the_embeddings(controller):
    """CLIP and DINOv2 vectors live in different spaces; comparing across them
    yields clusters that look plausible and mean nothing."""
    controller.embeddings = {"a": np.array([1.0], dtype=np.float32)}
    assert controller.set_model("DINOv2 (base)") is True
    assert controller.embeddings == {}
    assert controller.model_name == "DINOv2 (base)"


def test_reselecting_the_same_model_changes_nothing(controller):
    kept = {"a": np.array([1.0], dtype=np.float32)}
    controller.embeddings = kept
    assert controller.set_model(controller.model_name) is False
    assert controller.embeddings is kept


def test_an_unknown_model_is_refused(controller):
    kept = {"a": np.array([1.0], dtype=np.float32)}
    controller.embeddings = kept
    assert controller.set_model("ResNet from a dream") is False
    assert controller.embeddings is kept


def test_every_offered_model_can_be_selected(controller):
    """The picker is populated from `available_models`; an entry it offers but
    `set_model` rejects would be a dead menu item."""
    for name in controller.available_models():
        controller.set_model(name)
        assert controller.model_name == name


# --- cooperation with the #71 review ranking -------------------------------


# --- seeding the split (#80 question 1) ------------------------------------


def test_without_a_curation_run_the_grouping_is_purely_structural(controller):
    """And nothing is computed. An earlier attempt clustered on demand from the
    export path -- a pure-Python O(n^2) sweep on the GUI thread, 43 seconds at
    800 images, run even at a 0% validation split where the result was thrown
    away."""
    controller.mw.image_slices["clip"] = _Slices(["clip_F00000", "clip_F00001"])
    controller.clusters = lambda *a, **k: pytest.fail("clustered with no run")

    groups = controller.split_groups(["clip_F00000", "clip_F00001", "other.png"])

    assert groups["clip_F00000"] == groups["clip_F00001"] == "clip"
    assert groups["other.png"] == "other.png"


def test_clusters_group_frames_that_were_extracted_as_separate_files(controller):
    """The case ADR-044 left open: the dot in `clip_F00001.png` says
    "independent file" and the pixels say otherwise. Structure cannot see it;
    the embeddings can."""
    names = ["clip_F00001.png", "clip_F00002.png", "unrelated.png"]
    controller.embeddings = {
        "clip_F00001.png": np.array([1.0, 0.0], dtype=np.float32),
        "clip_F00002.png": np.array([1.0, 0.01], dtype=np.float32),
        "unrelated.png": np.array([0.0, 1.0], dtype=np.float32),
    }

    groups = controller.split_groups(names)

    assert groups["clip_F00001.png"] == groups["clip_F00002.png"]
    assert groups["unrelated.png"] != groups["clip_F00001.png"]


def test_refinement_translates_clusters_for_a_keyed_split(controller):
    """The SAM path splits on "{index}:{name}", so untranslated clusters would
    match nothing -- in silence."""
    controller.embeddings = {
        "a.png": np.array([1.0, 0.0], dtype=np.float32),
        "b.png": np.array([1.0, 0.01], dtype=np.float32),
    }
    keyed_groups = {"0:a.png": "a.png", "1:b.png": "b.png"}

    refined = controller.refine(
        keyed_groups, {"0:a.png": "a.png", "1:b.png": "b.png"}
    )

    assert refined["0:a.png"] == refined["1:b.png"]


def test_refinement_without_embeddings_returns_the_grouping_unchanged(controller):
    groups = {"0:a": "a", "1:b": "b"}
    assert controller.refine(groups, {"0:a": "a", "1:b": "b"}) == groups


def test_the_clusters_are_computed_once_per_embedding_set(controller):
    """Every export and every training launch asks for them, and one pass is
    several seconds at the supported ceiling. Cheap once is not cheap four
    times."""
    controller.embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.01], dtype=np.float32),
    }
    passes = []
    import src.digitalsreeni_image_annotator.controllers.curation_controller as module

    real = module.similarity.cluster

    def _counted(embeddings, threshold):
        passes.append(threshold)
        return real(embeddings, threshold)

    controller.clusters()
    module.similarity.cluster = _counted
    try:
        controller.clusters()
        controller.clusters()
        assert passes == [], "the clusters were recomputed"

        # New vectors invalidate it — a memo that outlived them would describe
        # the wrong dataset.
        controller.embeddings = dict(controller.embeddings)
        controller.clusters()
        assert len(passes) == 1

        # So does a different threshold.
        controller.clusters(0.5)
        assert len(passes) == 2
    finally:
        module.similarity.cluster = real


def test_the_backend_cannot_be_switched_while_a_run_is_in_flight(controller):
    """The invariant belongs on the controller, not only on the one view that
    respects it. Switching mid-run leaves `model_name` on the new backend while
    the running loop writes the OLD backend's vectors under new-backend cache
    keys -- into a file that outlives the session."""
    controller._computing = True
    assert controller.set_model("DINOv2 (base)") is False
    assert controller.model_name != "DINOv2 (base)"


def test_embeddings_are_stored_as_float32_arrays(controller, monkeypatch):
    """At the supported ceiling this is 60 MB against roughly 500 MB of Python
    float objects, which is the claim the ADR rests on.

    Stubbed at `_embed_one`, deliberately: letting the real one run on a fake
    file yields no embeddings, and `compute` then opens a modal QMessageBox
    with nobody to dismiss it -- so the test would hang rather than fail.
    """
    controller.mw.all_images.append({"file_name": "clip.mp4"})
    controller.mw.image_slices["clip"] = _Slices(["clip_F00000", "clip_F00001"])
    monkeypatch.setattr(
        controller, "_embed_one", lambda _kind, _payload, _cache: ([0.1, 0.2], False)
    )

    assert controller.compute() is True

    stored = next(iter(controller.embeddings.values()))
    assert isinstance(stored, np.ndarray)
    assert stored.dtype == np.float32


def test_a_cancelled_run_still_saves_what_it_computed(controller, monkeypatch):
    """Embedding is the expensive half; throwing away a cancelled run's work
    would make the next attempt start from nothing."""
    controller.mw.all_images.append({"file_name": "clip.mp4"})
    controller.mw.image_slices["clip"] = _Slices(["clip_F00000", "clip_F00001"])
    saved = []
    real_cache = controller.cache()
    monkeypatch.setattr(real_cache, "save", lambda: saved.append(True))

    from PyQt6.QtWidgets import QProgressDialog

    monkeypatch.setattr(QProgressDialog, "wasCanceled", lambda _self: True)

    assert controller.compute() is False
    assert saved, "a cancelled run discarded the embeddings it had computed"


def test_a_second_run_cannot_start_while_one_is_in_flight(controller, monkeypatch):
    """`compute` spins `processEvents` on every item behind a NON-modal
    progress dialog, so the backend combo stays live. Re-entering unloads the
    model the outer loop is still using and leaves a mixed CLIP+DINOv2
    embedding set — undetectable downstream, since both are 768-d, and
    `refine` feeds it into a real training run's split (ADR-013).

    `_compute` is stubbed rather than left to run: without the guard it reaches
    a modal QMessageBox with nobody to dismiss it, so the test would announce
    the bug by hanging the suite forever instead of failing. A mutation gate
    caught that — it had to kill the run on a timeout.
    """
    monkeypatch.setattr(
        controller, "_compute", lambda _parent: pytest.fail("re-entered a run")
    )
    controller._computing = True
    assert controller.compute() is False


def test_the_guard_is_released_even_when_a_run_fails(controller, monkeypatch):
    def _boom(_parent):
        raise RuntimeError("embedding exploded")

    monkeypatch.setattr(controller, "_compute", _boom)
    with pytest.raises(RuntimeError):
        controller.compute()
    assert controller.is_computing() is False


# --- cooperation with the review ranking (#71) -----------------------------


def _uncertainty(score):
    return {"score": score, "mode": MODE_UNCERTAINTY}


def test_without_review_scores_the_suggestion_is_the_most_typical(controller):
    controller.embeddings = {
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "middle": np.array([1.0, 0.2], dtype=np.float32),
        "right": np.array([1.0, 0.4], dtype=np.float32),
    }
    assert controller.suggested(["left", "middle", "right"]) == (
        "middle",
        "most typical",
    )


def test_with_uncertainty_scores_the_suggestion_is_the_most_uncertain(controller):
    """Precedence, not a combined score: redundancy says these three are
    interchangeable, uncertainty says which one is worth annotating."""
    controller.embeddings = {
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "middle": np.array([1.0, 0.2], dtype=np.float32),
        "right": np.array([1.0, 0.4], dtype=np.float32),
    }
    controller.mw.review_controller = _Review({
        "left": _uncertainty(0.2),
        "middle": _uncertainty(0.4),
        "right": _uncertainty(0.9),
    })
    assert controller.suggested(["left", "middle", "right"]) == (
        "right",
        "most uncertain",
    )


def test_mixed_score_kinds_fall_back_to_the_medoid(controller):
    """Disagreement is measured against labels, uncertainty against nothing.
    Ranking a cluster that mixes them compares two different quantities."""
    controller.embeddings = {
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "middle": np.array([1.0, 0.2], dtype=np.float32),
        "right": np.array([1.0, 0.4], dtype=np.float32),
    }
    controller.mw.review_controller = _Review({
        "left": _uncertainty(0.2),
        "middle": {"score": 9.9, "mode": MODE_DISAGREEMENT},
        "right": _uncertainty(0.9),
    })
    assert controller.suggested(["left", "middle", "right"]) == (
        "middle",
        "most typical",
    )


def test_a_partly_scored_cluster_falls_back_to_the_medoid(controller):
    """Otherwise the members that happen to have been measured outrank the ones
    that were not, for no reason at all."""
    controller.embeddings = {
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "middle": np.array([1.0, 0.2], dtype=np.float32),
        "right": np.array([1.0, 0.4], dtype=np.float32),
    }
    controller.mw.review_controller = _Review({"right": _uncertainty(0.9)})
    assert controller.suggested(["left", "middle", "right"]) == (
        "middle",
        "most typical",
    )


def test_a_record_without_a_score_still_counts_as_unmeasured(controller):
    """The coverage check is not redundant with the mode check.

    Every *ordinary* record carries both a score and a mode, so "all members
    are uncertainty-mode" normally implies "all members are scored" -- which is
    why dropping the count check survived a first round of tests. A record with
    a mode and no score separates them, and without the count check the ranking
    walks straight into a KeyError.
    """
    controller.embeddings = {
        "left": np.array([1.0, 0.0], dtype=np.float32),
        "middle": np.array([1.0, 0.2], dtype=np.float32),
        "right": np.array([1.0, 0.4], dtype=np.float32),
    }
    controller.mw.review_controller = _Review({
        "left": {"score": None, "mode": MODE_UNCERTAINTY},
        "middle": _uncertainty(0.4),
        "right": _uncertainty(0.9),
    })
    assert controller.suggested(["left", "middle", "right"]) == (
        "middle",
        "most typical",
    )


def test_review_scores_are_empty_without_a_review_controller(controller):
    assert controller.review_scores(["a", "b"]) == {}
    assert controller.suggested([]) == (None, "")


def test_ties_in_uncertainty_are_broken_deterministically(controller):
    controller.embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
    }
    controller.mw.review_controller = _Review({
        "a": _uncertainty(0.5), "b": _uncertainty(0.5),
    })
    assert controller.suggested(["a", "b"]) == controller.suggested(["b", "a"])
