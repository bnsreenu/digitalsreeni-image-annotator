"""Image embeddings for dataset curation (issue #72).

One vector per image, so near-duplicates and coverage gaps become visible. The
clustering that consumes these lives in :mod:`core.similarity` and is Qt-free
and model-free; this file is the part that owns a model.

**Backend choice: CLIP, behind an interface.** ``openai/clip-vit-base-patch32``
is small, widely cached and well understood, and its image tower alone is
enough here — the text tower is unused. DINOv2 (``facebook/dinov2-base``) is
generally stronger on pure visual similarity, particularly for texture-heavy
microscopy where CLIP's semantic bias is unhelpful. Rather than guess, the
backend is a name plus a loader, so DINOv2 can be swapped in and compared
without touching a line of clustering code. CLIP is the default for footprint
and familiarity.

**The cache is what makes this usable twice.** Embeddings are keyed by content
hash *and* model identity and persisted beside the project; a second run over
an unchanged dataset is nearly instant. Without that nobody runs it a second
time, and a curation tool you run once is a curation tool you do not use.

Nothing here ever deletes or modifies an image. Selection and recommendation
only — removing data on a similarity heuristic is not recoverable and is not
this feature's call.
"""

import gc
import hashlib
import json
import os

from PyQt6.QtCore import QObject

from ..core.logging_config import get_logger
from ..core.similarity import l2_normalise

logger = get_logger(__name__)

# name -> (hugging-face repo id, kind). `kind` picks the loading path, since
# CLIP needs its vision tower addressed explicitly while DINOv2 is a plain
# feature extractor.
EMBEDDING_MODELS = {
    "CLIP (ViT-B/32)": ("openai/clip-vit-base-patch32", "clip"),
    "DINOv2 (base)": ("facebook/dinov2-base", "dinov2"),
}
DEFAULT_MODEL = "CLIP (ViT-B/32)"

# Embeddings are computed at reduced resolution: these models resize to 224 px
# internally anyway, so feeding full-resolution microscopy costs decode time
# for no additional signal.
EMBED_SIZE = 224

CACHE_FILENAME = ".embedding_cache.json"


class EmbeddingUnavailableError(RuntimeError):
    """The backend could not be loaded (not downloaded, no network, no torch).

    Raised rather than returning None so the UI boundary can report something
    actionable instead of a silent empty result (ADR-031).
    """


def content_hash(path, chunk_size=1 << 20):
    """Stable digest of a file's bytes, or None if it cannot be read.

    Content, not mtime: copying a project or touching a file must not
    invalidate a perfectly good embedding, and editing an image must.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        logger.warning("could not hash %s", path)
        return None


class EmbeddingCache:
    """Content-hash-keyed embedding store, persisted as JSON beside the project.

    Keyed by ``(model_name, content_hash)`` so switching backend does not
    silently reuse the other one's vectors — which would produce clusters that
    look plausible and mean nothing.
    """

    def __init__(self, directory=None):
        self.directory = directory
        self._entries = {}
        if directory:
            self.load()

    @property
    def path(self):
        return os.path.join(self.directory, CACHE_FILENAME) if self.directory else None

    @staticmethod
    def _key(model_name, digest):
        return f"{model_name}::{digest}"

    def get(self, model_name, digest):
        if digest is None:
            return None
        return self._entries.get(self._key(model_name, digest))

    def put(self, model_name, digest, vector):
        if digest is None:
            return
        self._entries[self._key(model_name, digest)] = list(vector)

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self._entries = {
                    key: value for key, value in data.items() if isinstance(value, list)
                }
        except (OSError, ValueError):
            # A corrupt cache is a performance problem, never a correctness
            # one -- drop it and recompute rather than failing the run.
            logger.warning("discarding unreadable embedding cache at %s", self.path)
            self._entries = {}

    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self._entries, handle)
        except OSError:
            logger.warning("could not write the embedding cache to %s", self.path)

    def __len__(self):
        return len(self._entries)


class EmbeddingUtils(QObject):
    """Loads an embedding backend and produces one vector per image.

    Mirrors :class:`DINOUtils`: lazy import (ADR-012/016) so a missing or
    undownloaded model never blocks startup, device resolution shared with the
    rest of the inference layer, and a real ``unload`` that follows the GPU
    reclaim discipline.
    """

    def __init__(self):
        super().__init__()
        self._model = None
        self._processor = None
        self._device = None
        self.model_name = None

    # --- model lifecycle ---

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_name=DEFAULT_MODEL):
        """Load ``model_name``, downloading it on first use.

        Raises :class:`EmbeddingUnavailableError` with an actionable message
        rather than failing silently — a curation run that quietly produced no
        clusters would look like "no duplicates found".
        """
        if self._model is not None and self.model_name == model_name:
            return
        if model_name not in EMBEDDING_MODELS:
            raise EmbeddingUnavailableError(f"Unknown embedding model: {model_name}")
        repo_id, kind = EMBEDDING_MODELS[model_name]

        try:
            from ..core.torch_utils import resolve_torch_device

            device, _ = resolve_torch_device()
            if kind == "clip":
                from transformers import CLIPImageProcessor, CLIPVisionModel

                self._processor = CLIPImageProcessor.from_pretrained(repo_id)
                self._model = CLIPVisionModel.from_pretrained(repo_id).to(device)
            else:
                from transformers import AutoImageProcessor, AutoModel

                self._processor = AutoImageProcessor.from_pretrained(repo_id)
                self._model = AutoModel.from_pretrained(repo_id).to(device)
            self._model.eval()
            self._device = device
            self.model_name = model_name
            logger.info("embedding model %s loaded on %s", model_name, device)
        except Exception as exc:
            self._model = self._processor = None
            self.model_name = None
            raise EmbeddingUnavailableError(
                f"Could not load '{model_name}' ({repo_id}). It downloads on "
                "first use, so this usually means no network connection or no "
                f"disk space. Original error: {exc}"
            ) from exc

    def unload(self):
        """Release the model and reclaim GPU memory.

        Follows the documented order (CLAUDE.md, "Releasing Model GPU Memory"):
        move to CPU, drop references, collect, then clear the CUDA caches.
        Setting references to None alone leaves circular refs pinned and shows
        no drop at all in Task Manager.
        """
        model = self._model
        self._model = None
        self._processor = None
        self._device = None
        self.model_name = None
        if model is not None:
            try:
                model.cpu()
            except Exception:
                logger.debug("embedding model could not be moved to CPU", exc_info=True)
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.debug("CUDA cache clear skipped", exc_info=True)

    # --- embedding ---

    def embed_qimage(self, qimage):
        """Unit-length embedding of one ``QImage``.

        Goes through the same 8-bit RGB conversion the display path uses, so a
        16-bit or grayscale microscopy image is normalised the way it is shown
        (ADR-010) before hitting a model trained on natural RGB photographs.
        Feeding raw 16-bit values would produce a technically-valid embedding
        of a nearly-black image.
        """
        if self._model is None:
            raise EmbeddingUnavailableError("No embedding model is loaded.")

        import numpy as np
        import torch
        from PIL import Image

        from .sam_utils import _qimage_to_numpy

        array = _qimage_to_numpy(qimage)
        pil = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        pil = pil.resize((EMBED_SIZE, EMBED_SIZE))

        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=1)
        vector = pooled.squeeze(0).cpu().numpy().tolist()
        return l2_normalise(vector)
