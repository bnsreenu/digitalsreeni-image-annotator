"""Lazy, bounded materialisation of multi-dimensional image slices (issue #45).

Multi-dim TIFF/CZI stacks used to be exploded eagerly: every slice was
converted to an RGB888 ``QImage`` up front and all of them were held for the
whole session (``image_slices[base] = [(name, qimage), ...]``). A large 5D
stack could be several GB of live QImages.

This module keeps the identical ``(name, qimage)`` interface the rest of the
app relies on, but materialises each slice's QImage **on demand** and holds at
most :data:`LRU_CAPACITY` of them process-wide.

Strategy A (see ADR): the already-decoded source ndarray from
``tif.asarray()`` / CZI is *retained* by a :class:`SliceProvider`, and each
slice's QImage is reconstructed lazily through the exact ADR-010 8-bit-RGB
pipeline (``image_utils.convert_to_8bit_rgb`` / ``normalize_array`` /
``array_to_qimage``) so lazy pixels are byte-identical to the old eager ones.
Full array-free (memmap/zarr) reading is a deliberate follow-up.

Public API (consumed by controllers and issue #47 video work):

- :data:`LRU_CAPACITY` — default per-process live-QImage budget.
- :class:`SliceLRU` + :func:`get_shared_lru` — the one shared cache.
- :func:`evict_prefix` / :func:`release_slices` — drop a stack's cached QImages.
- :func:`slice_names` — names of a slice collection with **no** pixel work.
- :class:`SliceProvider` — retains the source array, materialises one slice.
- :class:`LazySliceList` — the drop-in replacement for the old list of tuples.
"""

import collections

import numpy as np

from . import image_utils

# Default number of materialised slice QImages held live across ALL stacks.
# Shared, so total QImage memory stays bounded no matter how many stacks are
# open. Per-list this bounds each stack to ~this many live QImages too.
LRU_CAPACITY = 8


class SliceLRU:
    """Process-wide bounded LRU of materialised slice QImages.

    Keyed by ``(provider_id, slice_name)`` so every :class:`LazySliceList`
    shares one capacity budget. Least-recently-``get``/``put`` entry is
    evicted first once ``len`` exceeds ``capacity``.
    """

    def __init__(self, capacity=LRU_CAPACITY):
        self.capacity = capacity
        self._cache = collections.OrderedDict()

    def get(self, key):
        cache = self._cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def put(self, key, qimage):
        cache = self._cache
        cache[key] = qimage
        cache.move_to_end(key)
        # Capacity may have been lowered at runtime (tests) — evict to fit.
        while len(cache) > self.capacity:
            cache.popitem(last=False)

    def evict_prefix(self, provider_id):
        """Drop every entry belonging to ``provider_id`` (stack deletion)."""
        cache = self._cache
        for key in [k for k in cache if k[0] == provider_id]:
            del cache[key]

    def count_prefix(self, provider_id):
        """Number of cached entries for ``provider_id`` (test introspection)."""
        return sum(1 for k in self._cache if k[0] == provider_id)

    def clear(self):
        self._cache.clear()

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)


# The single process-wide cache shared by all LazySliceLists.
_SHARED_LRU = SliceLRU()


def get_shared_lru():
    """Return the process-wide :class:`SliceLRU` singleton."""
    return _SHARED_LRU


def evict_prefix(provider_id):
    """Evict every shared-LRU entry for ``provider_id``."""
    _SHARED_LRU.evict_prefix(provider_id)


def slice_names(slices):
    """Ordered slice names of ``slices`` **without materialising any QImage**.

    Accepts either a :class:`LazySliceList` (returns its precomputed
    ``names``) or a plain ``[(name, qimage), ...]`` list — several tests and
    legacy call sites still hand in the latter, and name-only scans (save,
    annotation-status, navigation) must never trigger pixel decoding.
    ``None`` / empty yields ``[]``.
    """
    names = getattr(slices, "names", None)
    if names is not None:
        return list(names)
    return [name for name, _ in (slices or [])]


def release_slices(slices):
    """Release a slice collection's cached QImages, if it is a LazySliceList.

    A no-op for plain-list slice collections (nothing to evict). Call this
    before dropping a stack from ``image_slices`` so its LRU entries don't
    linger.
    """
    release = getattr(slices, "release", None)
    if callable(release):
        release()


class SliceProvider:
    """Retains a loaded multi-dim source array; materialises one slice's
    QImage on demand.

    The ordered ``names`` list and the ``name -> full-index`` map are
    precomputed once (cheap, no pixel work) using the EXACT same logic the
    old eager ``ImageController.create_slices`` used, so slice naming stays
    byte-identical (it is the annotation key + export filename — any drift
    orphans annotations).
    """

    def __init__(self, image_array, dimensions, base_name):
        self._array = image_array
        self.dimensions = list(dimensions)
        self.base_name = base_name
        self.provider_id = id(self)
        self.names = []
        # name -> tuple(full_idx) for ND; name -> None for the 2D single slice.
        self._index_map = {}
        self._build_index()

    def _build_index(self):
        arr = self._array
        dims = self.dimensions
        base = self.base_name

        if arr.ndim == 2:
            self.names = [base]
            self._index_map = {base: None}
            return

        slice_indices = [i for i, dim in enumerate(dims) if dim not in ("H", "W")]
        for idx_tuple in np.ndindex(tuple(arr.shape[i] for i in slice_indices)):
            full_idx = [slice(None)] * len(dims)
            for i, val in zip(slice_indices, idx_tuple):
                full_idx[i] = val
            slice_name = f"{base}_" + "_".join(
                f"{dims[i]}{val + 1}" for i, val in zip(slice_indices, idx_tuple)
            )
            self.names.append(slice_name)
            self._index_map[slice_name] = tuple(full_idx)

    def extract(self, name):
        """Reconstruct ONE slice's QImage. Returns a FRESH QImage every call
        (never mutate a cached one — the SAM worker may be reading it,
        ADR-013). Unknown name -> ``None``."""
        if name not in self._index_map:
            return None
        full_idx = self._index_map[name]
        if full_idx is None:  # the 2D single-slice case
            normalized = image_utils.normalize_array(self._array)
            return image_utils.array_to_qimage(normalized)
        slice_array = self._array[full_idx]
        rgb_slice = image_utils.convert_to_8bit_rgb(slice_array)
        return image_utils.array_to_qimage(rgb_slice)


class LazySliceList:
    """Drop-in replacement for the old ``[(name, qimage), ...]`` slice list.

    Backed by a :class:`SliceProvider` and the shared :class:`SliceLRU`.
    Supports the access patterns every consumer uses:

    - ``lazy.get(name)`` — LRU-cached materialise (miss extracts + inserts).
    - ``lazy[i]`` -> ``(name, qimage)`` (probes like ``slices[0][1]``).
    - ``for name, qimage in lazy`` — one-at-a-time materialise, feeding/
      evicting the shared LRU so a full export/DINO-batch pass never holds
      more than ``capacity`` live in the cache.
    - ``len`` / ``bool`` / ``.names`` — name-only, no pixel work.
    - ``prefetch_around(name)`` — pin current +/-1 for instant Up/Down nav.
    - ``release()`` — drop this stack's cached QImages.
    """

    def __init__(self, provider):
        self.provider = provider
        self.names = provider.names
        self._lru = get_shared_lru()

    @property
    def provider_id(self):
        return self.provider.provider_id

    def get(self, name):
        """Return the slice QImage for ``name`` (LRU hit or fresh extract),
        or ``None`` for an unknown name."""
        key = (self.provider.provider_id, name)
        qimage = self._lru.get(key)
        if qimage is not None:
            return qimage
        qimage = self.provider.extract(name)
        if qimage is None:
            return None
        self._lru.put(key, qimage)
        return qimage

    def __getitem__(self, index):
        name = self.names[index]
        return (name, self.get(name))

    def __iter__(self):
        for name in self.names:
            yield (name, self.get(name))

    def __len__(self):
        return len(self.names)

    def __bool__(self):
        return bool(self.names)

    def prefetch_around(self, name):
        """Materialise ``name`` and its +/-1 neighbours (by index in
        ``names``) so navigation is instant. Synchronous, main-thread only."""
        try:
            idx = self.names.index(name)
        except ValueError:
            self.get(name)
            return
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(self.names):
                self.get(self.names[j])

    def release(self):
        """Evict this stack's entries from the shared LRU (on delete)."""
        self._lru.evict_prefix(self.provider.provider_id)
