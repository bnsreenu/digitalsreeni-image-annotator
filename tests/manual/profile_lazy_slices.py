"""Manual RSS profiling for lazy multi-dim slice loading (issue #45).

NOT collected by pytest (no ``test_`` prefix). Run it by hand to eyeball the
resident-set-size (RSS) win of retaining the source array + a bounded LRU of
materialised QImages instead of exploding every slice up front.

Usage (from the repo root, offscreen so no display is needed)::

    QT_QPA_PLATFORM=offscreen python tests/manual/profile_lazy_slices.py

It builds a synthetic 5D uint16 stack, measures RSS after (a) just building the
LazySliceList (names only, no pixels) and (b) touching every slice one at a
time through the LRU, and prints both. The gap between "eager-equivalent"
(N * H*W*3 bytes of live QImages) and the bounded LRU is the memory saved.

Requires ``psutil`` for RSS; falls back to a note if it is not installed.
"""

import numpy as np
from PyQt6.QtWidgets import QApplication

from digitalsreeni_image_annotator.core.slice_cache import (
    LRU_CAPACITY,
    LazySliceList,
    SliceProvider,
    get_shared_lru,
)


def _rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    app = QApplication.instance() or QApplication([])  # noqa: F841

    T, Z, C, H, W = 4, 8, 3, 1024, 1024  # 96 slices of ~3 MB RGB each
    arr = (np.random.rand(T, Z, C, H, W) * 65535).astype(np.uint16)

    base_rss = _rss_mb()
    provider = SliceProvider(arr, ["T", "Z", "C", "H", "W"], "profile")
    lazy = LazySliceList(provider)
    after_build = _rss_mb()

    # Touch every slice once (feeds/evicts the shared LRU).
    for name in lazy.names:
        _img = lazy.get(name)
    after_touch = _rss_mb()

    n = len(lazy.names)
    eager_qimage_mb = n * H * W * 3 / (1024 * 1024)

    print(f"slices: {n}  (LRU capacity {LRU_CAPACITY})")
    print(f"source array: {arr.nbytes / (1024 * 1024):.0f} MB (retained, Strategy A)")
    print(f"eager would hold ~{eager_qimage_mb:.0f} MB of live QImages")
    print(f"LRU holds at most {get_shared_lru().capacity} QImages "
          f"(~{get_shared_lru().capacity * H * W * 3 / (1024 * 1024):.0f} MB)")
    if base_rss is None:
        print("(install psutil for RSS numbers)")
    else:
        print(f"RSS: baseline {base_rss:.0f} MB -> after build {after_build:.0f} MB "
              f"-> after touching all slices {after_touch:.0f} MB")


if __name__ == "__main__":
    main()
