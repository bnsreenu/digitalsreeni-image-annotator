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

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from ..core import similarity
from ..core.logging_config import get_logger
from ..core.slice_cache import slice_names
from ..inference.embedding_utils import (
    DEFAULT_MODEL,
    EmbeddingCache,
    EmbeddingUnavailableError,
    EmbeddingUtils,
    content_hash,
)

logger = get_logger(__name__)


class CurationController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.embedder = EmbeddingUtils()
        self.model_name = DEFAULT_MODEL
        self.threshold = similarity.DEFAULT_SIMILARITY
        self.embeddings = {}
        self._cache = None

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
            slices = self.mw.image_slices.get(base_name)
            if slices:
                for name in slice_names(slices):
                    items.append((name, "slice", (slices, name)))
                continue
            path = self.mw.image_paths.get(file_name)
            if path and os.path.exists(path):
                items.append((file_name, "path", path))
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

    # --- run ---

    def run(self):
        """Embed everything, cluster, and show the report."""
        items = self.collect_work_items()
        if len(items) < 2:
            QMessageBox.information(
                self.mw,
                "Dataset similarity",
                "At least two images are needed to look for near-duplicates.",
            )
            return
        if len(items) > similarity.ALL_PAIRS_LIMIT:
            QMessageBox.warning(
                self.mw, "Dataset similarity", similarity.ALL_PAIRS_LIMIT_MESSAGE
            )
            return

        try:
            self.embedder.load(self.model_name)
        except EmbeddingUnavailableError as exc:
            QMessageBox.warning(self.mw, "Dataset similarity", str(exc))
            return

        cache = self.cache()
        progress = QProgressDialog(
            "Computing embeddings…", "Cancel", 0, len(items), self.mw
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
                return
            try:
                vector, from_cache = self._embed_one(kind, payload, cache)
            except EmbeddingUnavailableError as exc:
                progress.close()
                QMessageBox.warning(self.mw, "Dataset similarity", str(exc))
                return
            except Exception:
                logger.exception("embedding failed for %s", name)
                continue
            if vector is not None:
                embeddings[name] = vector
                cached_hits += int(from_cache)
        progress.setValue(len(items))
        cache.save()

        if len(embeddings) < 2:
            QMessageBox.information(
                self.mw,
                "Dataset similarity",
                "Not enough images could be embedded to compare.",
            )
            return

        self.embeddings = embeddings
        logger.info(
            "embedded %d image(s), %d from cache", len(embeddings), cached_hits
        )

        from ..dialogs.dataset_curation_dialog import DatasetCurationDialog

        DatasetCurationDialog(self.mw, self).exec()

    def _embed_one(self, kind, payload, cache):
        """``(vector, came_from_cache)`` for one work item.

        Only file-backed images are cacheable: a slice has no stable content
        hash of its own without decoding it, at which point the embedding is
        the cheap part.
        """
        if kind == "path":
            digest = content_hash(payload)
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

        slices, name = payload
        qimage = slices.get(name)
        if qimage is None:
            return None, False
        return self.embedder.embed_qimage(qimage), False

    # --- results ---

    def clusters(self, threshold=None):
        """Near-duplicate clusters at ``threshold``.

        Re-clustering is pure arithmetic over the embeddings already in memory,
        so moving the threshold slider is instant and never recomputes.
        """
        return similarity.cluster(
            self.embeddings, self.threshold if threshold is None else threshold
        )

    def representative(self, cluster_names):
        return similarity.representative(cluster_names, self.embeddings)

    def outliers(self, threshold=None):
        return similarity.outliers(
            self.embeddings, self.threshold if threshold is None else threshold
        )

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
