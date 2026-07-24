"""Unit tests for the lazy slice cache (issue #45).

`core/slice_cache.py` replaces the old "materialise every slice's QImage up
front and hold them for the session" behaviour with on-demand materialisation
behind a process-wide bounded LRU. These tests pin the invariants the rest of
the app depends on:

  1. slice NAMES are byte-identical to the old create_slices (`{base}_T1_Z5_C1`,
     1-based) — they are the annotation key + export filename;
  2. the shared LRU evicts beyond ``LRU_CAPACITY``;
  3. ``prefetch_around`` pins current +/-1 (instant Up/Down nav);
  4. iterating N names with a tiny capacity re-extracts but never holds more
     than ``capacity`` live in the LRU;
  5. lazily extracted pixels are byte-identical to the eager
     convert_to_8bit_rgb / array_to_qimage pipeline (ADR-010).
"""

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from digitalsreeni_image_annotator.core import image_utils
from digitalsreeni_image_annotator.core.slice_cache import (
    LRU_CAPACITY,
    LazySliceList,
    SliceProvider,
    get_shared_lru,
    release_slices,
    slice_names,
)


@pytest.fixture
def clean_lru():
    """Isolate each test from the process-wide shared LRU: clear it and
    restore its capacity afterwards (other tests share the singleton)."""
    cache = get_shared_lru()
    saved_capacity = cache.capacity
    cache.clear()
    yield cache
    cache.clear()
    cache.capacity = saved_capacity


def _spy_extract(provider):
    """Wrap ``provider.extract`` with a call counter; returns the counter dict."""
    calls = {"n": 0}
    original = provider.extract

    def counting(name):
        calls["n"] += 1
        return original(name)

    provider.extract = counting
    return calls


def _ramp(n, h=4, w=4):
    """A non-constant uint16 stack — each plane has varying values so the
    normalize step never divides by a zero (max == min) range."""
    return (np.arange(n * h * w, dtype=np.float64).reshape(n, h, w) % 60000).astype(
        np.uint16
    )


# ── (1) naming is byte-identical to create_slices ────────────────────────────

class TestNaming:
    def test_2d_single_slice_named_base(self, clean_lru):
        provider = SliceProvider(np.zeros((8, 6), np.uint16), ["H", "W"], "flat")
        assert provider.names == ["flat"]

    def test_3d_zhw_one_based(self, clean_lru):
        provider = SliceProvider(np.zeros((4, 8, 6), np.uint16), ["Z", "H", "W"], "stack3d")
        assert provider.names == [f"stack3d_Z{i}" for i in range(1, 5)]

    def test_5d_tzcyx_matches_ndindex_order_and_is_one_based(self, clean_lru):
        provider = SliceProvider(
            np.zeros((2, 5, 2, 8, 6), np.uint16),
            ["T", "Z", "C", "H", "W"],
            "stack5d",
        )
        expected = [
            f"stack5d_T{t + 1}_Z{z + 1}_C{c + 1}"
            for t in range(2)
            for z in range(5)
            for c in range(2)
        ]
        assert provider.names == expected
        assert provider.names[0] == "stack5d_T1_Z1_C1"
        assert "stack5d_T1_Z5_C1" in provider.names
        assert provider.names[-1] == "stack5d_T2_Z5_C2"
        assert len(provider.names) == 2 * 5 * 2

    def test_empty_leading_dim_yields_no_names(self, clean_lru):
        provider = SliceProvider(np.zeros((0, 4, 4), np.uint16), ["Z", "H", "W"], "e")
        assert provider.names == []


# ── (2) LRU eviction ─────────────────────────────────────────────────────────

class TestLRUEviction:
    def test_evicts_beyond_capacity_keeping_most_recent(self, clean_lru):
        clean_lru.capacity = LRU_CAPACITY
        n = LRU_CAPACITY + 4
        provider = SliceProvider(_ramp(n), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)

        for name in lazy.names:
            lazy.get(name)

        assert len(clean_lru) == LRU_CAPACITY
        retained = set(lazy.names[-LRU_CAPACITY:])
        for name in lazy.names:
            present = (provider.provider_id, name) in clean_lru
            assert present == (name in retained)

    def test_lru_hit_returns_same_object(self, clean_lru):
        provider = SliceProvider(_ramp(3), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)
        first = lazy.get(lazy.names[0])
        again = lazy.get(lazy.names[0])
        assert first is again  # cache hit, not a re-extract


# ── (3) prefetch pins current +/-1 ───────────────────────────────────────────

class TestPrefetch:
    def test_prefetch_around_pins_current_and_neighbors(self, clean_lru):
        clean_lru.capacity = 8
        provider = SliceProvider(_ramp(5), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)
        pid = provider.provider_id

        lazy.prefetch_around(lazy.names[2])

        assert (pid, lazy.names[1]) in clean_lru
        assert (pid, lazy.names[2]) in clean_lru
        assert (pid, lazy.names[3]) in clean_lru
        assert (pid, lazy.names[0]) not in clean_lru
        assert (pid, lazy.names[4]) not in clean_lru

    def test_prefetch_at_edges_stays_in_bounds(self, clean_lru):
        clean_lru.capacity = 8
        provider = SliceProvider(_ramp(3), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)
        pid = provider.provider_id

        lazy.prefetch_around(lazy.names[0])  # no left neighbour
        assert (pid, lazy.names[0]) in clean_lru
        assert (pid, lazy.names[1]) in clean_lru
        assert (pid, lazy.names[2]) not in clean_lru


# ── (4) iteration re-extracts but bounds the LRU ─────────────────────────────

class TestIterationBounded:
    def test_full_pass_reextracts_but_never_exceeds_capacity(self, clean_lru):
        clean_lru.capacity = 2
        provider = SliceProvider(_ramp(5), ["Z", "H", "W"], "s")
        calls = _spy_extract(provider)
        lazy = LazySliceList(provider)

        sizes = []
        for _name, _img in lazy:
            sizes.append(len(clean_lru))
        assert calls["n"] == 5           # one extract per name (all missed)
        assert max(sizes) <= 2           # never held more than capacity
        assert len(clean_lru) <= 2

        # A second pass re-extracts the evicted entries (nothing is permanently
        # pinned) and still never exceeds capacity.
        for _name, _img in lazy:
            assert len(clean_lru) <= 2
        assert calls["n"] == 10


# ── (5) lazy pixels == eager pipeline ────────────────────────────────────────

class TestPixelEquivalence:
    def test_extracted_slice_matches_eager_pipeline(self, clean_lru, qt_application):
        rng = np.random.RandomState(0)
        arr = (rng.rand(3, 8, 6) * 65535).astype(np.uint16)
        provider = SliceProvider(arr, ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)

        got = lazy.get("s_Z2")  # 1-based Z2 -> plane index 1

        eager_rgb = image_utils.convert_to_8bit_rgb(arr[1])
        eager = image_utils.array_to_qimage(eager_rgb)

        assert got.format() == eager.format() == QImage.Format.Format_RGB888
        assert (got.width(), got.height()) == (eager.width(), eager.height())
        for y in range(eager.height()):
            for x in range(eager.width()):
                assert got.pixel(x, y) == eager.pixel(x, y)

    def test_5d_tzcyx_slice_matches_eager_pipeline(self, clean_lru, qt_application):
        # Pixel-equality on the 5D dimension order that shipped the historical
        # [-ndim:] axis-slice bug, so byte-identity is proven, not just argued.
        rng = np.random.RandomState(2)
        arr = (rng.rand(2, 3, 2, 8, 6) * 65535).astype(np.uint16)
        dims = ["T", "Z", "C", "H", "W"]
        provider = SliceProvider(arr, dims, "vol")
        lazy = LazySliceList(provider)

        # A mid-stack slice: T2_Z2_C1 -> array indices [1, 1, 0].
        name = "vol_T2_Z2_C1"
        assert name in lazy.names
        got = lazy.get(name)
        eager_rgb = image_utils.convert_to_8bit_rgb(arr[1, 1, 0])
        eager = image_utils.array_to_qimage(eager_rgb)

        assert got.format() == eager.format() == QImage.Format.Format_RGB888
        assert (got.width(), got.height()) == (eager.width(), eager.height())
        for y in range(eager.height()):
            for x in range(eager.width()):
                assert got.pixel(x, y) == eager.pixel(x, y)

    def test_2d_slice_matches_grayscale_pipeline(self, clean_lru, qt_application):
        rng = np.random.RandomState(1)
        arr = (rng.rand(8, 6) * 65535).astype(np.uint16)
        provider = SliceProvider(arr, ["H", "W"], "flat")
        lazy = LazySliceList(provider)

        got = lazy.get("flat")
        eager = image_utils.array_to_qimage(image_utils.normalize_array(arr))

        assert got.format() == eager.format() == QImage.Format.Format_Grayscale8
        for y in range(eager.height()):
            for x in range(eager.width()):
                assert got.pixel(x, y) == eager.pixel(x, y)


# ── list protocol + helpers ──────────────────────────────────────────────────

class TestListProtocol:
    def test_getitem_iter_len_bool(self, clean_lru):
        provider = SliceProvider(_ramp(3), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)

        assert len(lazy) == 3
        assert bool(lazy) is True

        name0, img0 = lazy[0]
        assert name0 == "s_Z1"
        assert isinstance(img0, QImage)

        pairs = list(lazy)
        assert [n for n, _ in pairs] == lazy.names
        assert all(isinstance(im, QImage) for _, im in pairs)

        assert lazy.get("does-not-exist") is None

    def test_empty_list_is_falsy(self, clean_lru):
        provider = SliceProvider(np.zeros((0, 4, 4), np.uint16), ["Z", "H", "W"], "e")
        lazy = LazySliceList(provider)
        assert len(lazy) == 0
        assert bool(lazy) is False
        assert list(lazy) == []


class TestSliceNamesHelper:
    def test_accepts_lazy_plain_list_and_none(self, clean_lru):
        provider = SliceProvider(_ramp(3), ["Z", "H", "W"], "s")
        lazy = LazySliceList(provider)

        assert slice_names(lazy) == lazy.names
        assert slice_names([("a", None), ("b", None)]) == ["a", "b"]
        assert slice_names(None) == []
        assert slice_names([]) == []

    def test_does_not_materialise_pixels(self, clean_lru):
        provider = SliceProvider(_ramp(3), ["Z", "H", "W"], "s")
        calls = _spy_extract(provider)
        lazy = LazySliceList(provider)

        slice_names(lazy)
        assert calls["n"] == 0


class TestReleaseEviction:
    def test_release_evicts_only_its_own_entries(self, clean_lru):
        arr = _ramp(3)
        p1 = SliceProvider(arr, ["Z", "H", "W"], "a")
        p2 = SliceProvider(arr, ["Z", "H", "W"], "b")
        l1, l2 = LazySliceList(p1), LazySliceList(p2)
        l1.get(l1.names[0])
        l2.get(l2.names[0])

        assert clean_lru.count_prefix(p1.provider_id) == 1
        assert clean_lru.count_prefix(p2.provider_id) == 1

        l1.release()
        assert clean_lru.count_prefix(p1.provider_id) == 0
        assert clean_lru.count_prefix(p2.provider_id) == 1  # untouched

        release_slices(l2)  # module-level helper
        assert clean_lru.count_prefix(p2.provider_id) == 0

    def test_release_slices_no_op_on_plain_list(self, clean_lru):
        release_slices([("a", None)])  # must not raise
        release_slices(None)
