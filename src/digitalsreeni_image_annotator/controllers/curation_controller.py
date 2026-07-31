"""Embedding-based dataset curation (issue #72).

Computes one embedding per image (including slices and video frames — the most
redundant data the app handles, and therefore the primary use case), clusters
near-duplicates, and reports what could be skipped.

**Never deletes anything.** Selection and recommendation only. Removing data on
a similarity heuristic is not recoverable and is not this feature's call — the
controller has no delete path at all, which is the strongest form that promise
can take.
"""

import os

import numpy as np
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..core import similarity
from ..core.dataset_split import derive_groups, merge_groups, translate_clusters
from ..core.logging_config import get_logger
from ..core.slice_cache import slice_names
from ..inference.embedding_utils import (
    DEFAULT_MODEL,
    EMBEDDING_MODELS,
    EmbeddingCache,
    EmbeddingUnavailableError,
    EmbeddingUtils,
    content_hash,
    slice_digest,
)
from .review_controller import MODE_UNCERTAINTY

logger = get_logger(__name__)


class CurationController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.embedder = EmbeddingUtils()
        self.model_name = DEFAULT_MODEL
        self.threshold = similarity.DEFAULT_SIMILARITY
        self.mode_threshold = similarity.MODE_SIMILARITY
        self._embeddings = {}
        self._cache = None
        # In-flight guard (ADR-013). `compute` spins `processEvents` on every
        # item and its progress dialog is non-modal, so the model combo stays
        # live for the whole run: a second selection re-enters here, unloads
        # the model the outer loop is still using, and the outer loop then
        # overwrites `embeddings` with a mixture of CLIP and DINOv2 vectors.
        # Both are 768-d, so nothing downstream can detect it -- and `refine`
        # feeds those clusters straight into a real training run's split.
        self._computing = False
        # Cleared on every assignment to `embeddings` (see the property below),
        # so derived results can be cached without a stale-read risk.
        self._clusters_cache = None
        # Source-file digests for this run. Hashing a 2 GB video once is fine;
        # hashing it once per frame is not, and the frames are the reason the
        # cache exists at all.
        self._digests = {}

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        """A property purely so every assignment drops the derived cache.

        The dialog restores a previous model's vectors by plain assignment and
        `set_model` clears them, so assignment is the invalidation point.
        Mutating the returned dict **in place** is not supported and would not
        invalidate anything -- every producer here replaces it wholesale.
        """
        self._embeddings = value
        self._clusters_cache = None

    def is_computing(self):
        return self._computing

    # --- collection ---

    def collect_work_items(self):
        """``[(name, kind, payload)]`` for everything embeddable.

        ``kind`` is ``"path"`` for a plain image on disk or ``"slice"`` for a
        stack slice / video frame, whose pixels come from the lazy slice list
        rather than a file. Enumerating slices is the point: consecutive video
        frames are the most redundant data the app handles, and iterating
        ``all_images`` alone would miss every one of them.
        """
        items = []
        for info in self.mw.all_images:
            file_name = info.get("file_name")
            if not file_name:
                continue
            base_name = os.path.splitext(file_name)[0]
            source = self.mw.image_paths.get(file_name)
            slices = self.mw.image_slices.get(base_name)
            if slices:
                # The source path travels with each slice so its digest can key
                # the cache. A stack with no path entry still embeds; it just
                # cannot be cached.
                for name in slice_names(slices):
                    items.append((name, "slice", (slices, name, source)))
                continue
            if source and os.path.exists(source):
                items.append((file_name, "path", source))
        return items

    def cache(self):
        """The embedding cache, stored beside the project file.

        A project-less session gets an in-memory cache: still useful within the
        run, and writing a cache next to an unrelated working directory would
        be litter.
        """
        if self._cache is None:
            project_file = getattr(self.mw, "current_project_file", None)
            directory = os.path.dirname(project_file) if project_file else None
            self._cache = EmbeddingCache(directory)
        return self._cache

    def _source_digest(self, path):
        """Content hash of ``path``, computed at most once per run."""
        if path not in self._digests:
            self._digests[path] = content_hash(path) if path else None
        return self._digests[path]

    # --- model selection ---

    def available_models(self):
        return list(EMBEDDING_MODELS)

    def set_model(self, model_name):
        """Switch backend. Returns True if anything changed.

        The embeddings are dropped rather than kept: CLIP and DINOv2 vectors
        live in different spaces, and comparing one to the other produces
        clusters that look entirely plausible and mean nothing. The persisted
        cache is keyed by model identity, so switching back costs nothing.

        Which of the two is better is a per-dataset question -- DINOv2 is
        generally stronger on pure visual similarity, CLIP carries semantic
        bias that helps on natural photographs and hurts on texture-heavy
        microscopy. Rather than answer it globally, this makes it answerable.
        """
        if model_name == self.model_name or model_name not in EMBEDDING_MODELS:
            return False
        if self._computing:
            # The invariant belongs here, not only on the one view that
            # respects it: switching mid-run would leave `model_name` pointing
            # at the new backend while the running loop writes the *old*
            # backend's vectors under new-backend cache keys -- persistently.
            logger.warning("model switch ignored: a run is in flight")
            return False
        # Reclaim the old model's VRAM before the new one is loaded, rather
        # than leaving two on the card until the garbage collector notices.
        self.embedder.unload()
        self.model_name = model_name
        self.embeddings = {}
        return True

    # --- run ---

    def run(self):
        """Embed everything, cluster, and show the report."""
        if not self.compute():
            return

        from ..dialogs.dataset_curation_dialog import DatasetCurationDialog

        DatasetCurationDialog(self.mw, self).exec()

    def compute(self, parent=None):
        """Embed every work item. True when there is enough to analyse.

        Split out of :meth:`run` so the dialog can re-embed in place when the
        user switches backend, instead of closing and reopening itself.
        """
        parent = parent or self.mw
        if self._computing:
            logger.warning("curation run re-entered; ignoring")
            return False
        self._computing = True
        try:
            return self._compute(parent)
        finally:
            self._computing = False

    def _compute(self, parent):
        self._digests = {}
        items = self.collect_work_items()
        if len(items) < 2:
            QMessageBox.information(
                parent,
                "Dataset similarity",
                "At least two images are needed to look for near-duplicates.",
            )
            return False
        if len(items) > similarity.ALL_PAIRS_LIMIT:
            QMessageBox.warning(
                parent, "Dataset similarity", similarity.ALL_PAIRS_LIMIT_MESSAGE
            )
            return False

        try:
            self.embedder.load(self.model_name)
        except EmbeddingUnavailableError as exc:
            QMessageBox.warning(parent, "Dataset similarity", str(exc))
            return False

        cache = self.cache()
        progress = QProgressDialog(
            f"Computing {self.model_name} embeddings…",
            "Cancel",
            0,
            len(items),
            parent,
        )
        progress.setWindowTitle("Dataset similarity")
        progress.setMinimumDuration(0)

        embeddings = {}
        cached_hits = 0
        for index, (name, kind, payload) in enumerate(items):
            progress.setValue(index)
            progress.setLabelText(f"Embedding {name}…")
            QApplication.processEvents()
            if progress.wasCanceled():
                logger.info("curation run cancelled after %d image(s)", index)
                progress.close()
                # Whatever was computed is still in the cache and will be
                # reused; it is simply not enough to report on.
                cache.save()
                return False
            try:
                vector, from_cache = self._embed_one(kind, payload, cache)
            except EmbeddingUnavailableError as exc:
                progress.close()
                QMessageBox.warning(parent, "Dataset similarity", str(exc))
                return False
            except Exception:
                logger.exception("embedding failed for %s", name)
                continue
            if vector is not None:
                # float32 arrays, not Python float lists: at the supported
                # ceiling that is the difference between roughly 60 MB and
                # half a gigabyte of live objects, and every consumer in
                # core.similarity works on arrays anyway.
                embeddings[name] = np.asarray(vector, dtype=np.float32)
                cached_hits += int(from_cache)
        progress.setValue(len(items))
        cache.save()

        if len(embeddings) < 2:
            QMessageBox.information(
                parent,
                "Dataset similarity",
                "Not enough images could be embedded to compare.",
            )
            return False

        self.embeddings = embeddings
        logger.info(
            "embedded %d image(s) with %s, %d from cache",
            len(embeddings),
            self.model_name,
            cached_hits,
        )
        return True

    def _embed_one(self, kind, payload, cache):
        """``(vector, came_from_cache)`` for one work item.

        Slices are cached too (#82). They used not to be, on the reasoning that
        a slice has no content hash of its own -- but the source file does, and
        the slice name is the coordinate within it, so the pair is a stable key
        that costs no decoding (:func:`slice_digest`). Since slices and video
        frames are the *primary* case for this feature, "everything except the
        primary case is cached" meant a second run was never fast.
        """
        if kind == "path":
            digest = self._source_digest(payload)
            cached = cache.get(self.model_name, digest)
            if cached is not None:
                return cached, True
            from PyQt6.QtGui import QImage

            qimage = QImage(payload)
            if qimage.isNull():
                return None, False
            vector = self.embedder.embed_qimage(qimage)
            cache.put(self.model_name, digest, vector)
            return vector, False

        slices, name, source = payload
        # The axis assignment is part of the key: two assignments of one array
        # can produce identical slice names for different pixels (see
        # `slice_digest`). A video provider has no `dimensions` and needs none.
        provider = getattr(slices, "provider", None)
        digest = slice_digest(
            self._source_digest(source), name, getattr(provider, "dimensions", None)
        )
        cached = cache.get(self.model_name, digest)
        if cached is not None:
            return cached, True
        qimage = slices.get(name)
        if qimage is None:
            return None, False
        vector = self.embedder.embed_qimage(qimage)
        cache.put(self.model_name, digest, vector)
        return vector, False

    # --- results ---

    def analyse(self, threshold=None):
        """``{"clusters", "outliers", "modes"}`` at ``threshold``.

        Re-analysing is arithmetic over vectors already in memory -- no
        inference, no file access -- and all three answers come from one sweep
        over the pairs, so the threshold slider costs a single pass.
        """
        return similarity.analyse(
            self.embeddings,
            self.threshold if threshold is None else threshold,
            self.mode_threshold,
        )

    def clusters(self, threshold=None):
        """Near-duplicate clusters at ``threshold``, memoised.

        The memo is what makes :meth:`refine` honest about its cost. Every
        export and every training launch asks for the clusters, and at the
        supported ceiling one pass is several seconds on the GUI thread —
        cheap once, not cheap four times. It is keyed by model and threshold,
        and dropped outright whenever ``embeddings`` is assigned, so it cannot
        outlive the vectors it describes.
        """
        threshold = self.threshold if threshold is None else threshold
        key = (self.model_name, threshold)
        if self._clusters_cache is not None and self._clusters_cache[0] == key:
            return self._clusters_cache[1]
        result = similarity.cluster(self.embeddings, threshold)
        self._clusters_cache = (key, result)
        return result

    def representative(self, cluster_names):
        return similarity.representative(cluster_names, self.embeddings)

    def cohesion(self, cluster_names):
        return similarity.cohesion(cluster_names, self.embeddings)

    def outliers(self, threshold=None):
        return similarity.outliers(
            self.embeddings, self.threshold if threshold is None else threshold
        )

    # --- seeding the train/val split (#80 question 1, ADR-044/045) ---

    def split_groups(self, names, image_slices=None):
        """``{name: group}`` for a split over ``names``, refined if possible.

        This is what the curation output is *for*. The report is the visible
        half; the useful half is that a near-duplicate cluster is evidence two
        images must not be split across train and val. Structure catches a
        stack's slices and a video's frames; clusters catch what structure
        cannot -- a folder of frames extracted as ordinary files, where the
        name says "independent image" and the pixels say otherwise.
        """
        groups = derive_groups(
            names,
            self.mw.image_slices if image_slices is None else image_slices,
        )
        return self.refine(groups)

    def refine(self, groups, names_by_key=None):
        """Fold this session's near-duplicate clusters into ``groups``.

        Returns ``groups`` untouched when no curation run has happened, and
        computes nothing in that case. That guard is the whole reason this is
        safe to call from an export path: an earlier attempt clustered the
        project *on demand* from the exporter, which meant a pure-Python
        O(n^2) sweep on the GUI thread on every export -- 43 seconds at 800
        images, and it ran even at a 0% validation split, where the result was
        thrown away.

        ``names_by_key`` maps split keys to image names for callers that do not
        split on names (the SAM path keys by ``"{index}:{name}"``).
        """
        if not self.embeddings:
            return groups
        # Seconds at the supported ceiling, on the GUI thread, immediately
        # after the user confirmed a dialog. Say so with the cursor rather than
        # looking frozen.
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            clusters = self.clusters()
        finally:
            QApplication.restoreOverrideCursor()
        if names_by_key is not None:
            clusters = translate_clusters(clusters, names_by_key)
        return merge_groups(groups, clusters)

    # --- cooperation with the review ranking (#71) ---

    def review_scores(self, names):
        """``{name: score}`` from the #71 review run, for the names it covers.

        Empty when no review has been run, and -- more often -- when the
        project is a video or a stack: review scoring runs from a file path and
        skips slices entirely, while curation exists mainly *for* slices. The
        two features overlap on plain-image projects and nowhere else, which is
        worth knowing before reading anything into an empty column.
        """
        review = getattr(self.mw, "review_controller", None)
        if review is None or not review.has_scores():
            return {}
        scored = {}
        for name in names or []:
            score = review.score_for(name)
            if score is not None:
                scored[name] = score
        return scored

    def suggested(self, cluster_names, scores=None):
        """``(name, reason)`` -- which member of a cluster to work on.

        **Precedence, not a combined score.** Redundancy decides *what can be
        skipped*; uncertainty decides *what is worth the effort* among what is
        left. Multiplying the two into one number would need a weight nobody
        can justify, and would hide which of the two drove the answer.

        So: with no review scores the answer is the medoid, the most typical
        frame -- the right pick when the question is which one to keep. With
        review scores it becomes the most uncertain member, the right pick when
        the question is which one to annotate.

        The scores must cover every member and all be uncertainty scores.
        Disagreement and uncertainty are different quantities on different
        scales (an annotated image is scored against its labels, an unannotated
        one against nothing), so ranking a cluster that mixes them would be
        comparing two different measurements -- and a half-covered cluster
        would rank the scored members above the unmeasured ones for no reason.
        """
        names = list(cluster_names or [])
        if not names:
            return None, ""
        if scores is None:
            scores = self.review_scores(names)
        else:
            # The dialog looks the whole report's scores up once; re-deriving
            # them per cluster is the same answer at N times the cost.
            scores = {name: scores[name] for name in names if name in scores}
        review = getattr(self.mw, "review_controller", None)
        uncertainty_only = review is not None and all(
            review.mode_for(name) == MODE_UNCERTAINTY for name in names
        )
        if len(scores) == len(names) and uncertainty_only:
            best = sorted(names, key=lambda name: (-scores[name], name))[0]
            return best, "most uncertain"
        return self.representative(names), "most typical"

    def select_in_image_list(self, names):
        """Select a cluster's images in the image list.

        The action a finding turns into. The image list is in
        ``ExtendedSelection`` mode so the group genuinely stays selected; the
        per-image context menu still operates on the item under the cursor, so
        this is a visual grouping and a starting point, not a bulk-operation
        target. Slice names have no row of their own and are skipped.
        """
        wanted = set(names or [])
        image_list = self.mw.image_list
        image_list.clearSelection()
        selected = 0
        for row in range(image_list.count()):
            item = image_list.item(row)
            if item is not None and item.text() in wanted:
                item.setSelected(True)
                selected += 1
        # Deliberately no setCurrentItem: currentRowChanged is wired to
        # switch_image, which would navigate away and collapse the very
        # multi-selection the user asked for.
        return selected

    def unload(self):
        self.embedder.unload()
