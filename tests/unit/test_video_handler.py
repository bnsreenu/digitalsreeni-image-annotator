"""Unit tests for the video-handler layer (issue #47).

These exercise ``core/video_handler.py`` headlessly — only cv2 + QImage are
involved, no main window. A tiny MJPG/.avi is written per test via the shared
``make_test_video`` fixture (tests/conftest.py). MJPG on a uniform fill
round-trips exactly, so frame ``i``'s red channel reads back as ``10*i`` and a
forgotten BGR→RGB conversion is caught by a blue-vs-red assertion.
"""

import pytest
from PyQt6.QtGui import QImage, qBlue, qGreen, qRed

from digitalsreeni_image_annotator.core.slice_cache import (
    LazySliceList,
    get_shared_lru,
)
from digitalsreeni_image_annotator.core.video_handler import (
    VIDEO_EXTS,
    VideoHandler,
    VideoSliceProvider,
    file_dialog_filter,
    frame_key,
    is_video,
    parse_frame_index,
)


# ── pure helpers (no video needed) ───────────────────────────────────────────

def test_frame_key_zero_padded():
    assert frame_key("v", 42) == "v_F00042"
    assert frame_key("clip", 0) == "clip_F00000"


def test_file_dialog_filter_covers_every_video_ext():
    """The open-dialog filter must offer every registered video container, so
    the app can't silently accept a format the file picker never shows (the
    #47 gap). Guards the VIDEO_EXTS-derived globs in file_dialog_filter():
    add an ext to VIDEO_EXTS and this fails until the filter covers it."""
    filt = file_dialog_filter()
    for ext in VIDEO_EXTS:
        assert f"*{ext}" in filt, f"{ext} missing from the open-dialog filter"
    # ...and it advertises videos, not just images.
    assert "Videos" in filt


def test_parse_frame_index_roundtrip():
    assert parse_frame_index("v_F00042") == 42
    assert parse_frame_index(frame_key("clip", 7)) == 7


def test_parse_frame_index_ignores_multidim_keys():
    # Anchored at END — a multi-dim slice key is NOT a frame key.
    assert parse_frame_index("stack_T1_Z5") is None
    assert parse_frame_index("plain") is None


def test_is_video_case_insensitive():
    assert is_video("clip.mp4")
    assert is_video("CLIP.MP4")
    assert is_video("movie.AVI")
    assert is_video("take.mov")
    assert not is_video("image.png")
    assert not is_video("stack.tif")


# ── VideoHandler ─────────────────────────────────────────────────────────────

def test_metadata(make_test_video, tmp_path):
    path = make_test_video(tmp_path, frames=8, width=32, height=24, fps=10.0)
    handler = VideoHandler(path)
    try:
        meta = handler.metadata()
        # The MJPG writer produces exactly 8 here; allow a codec that yields a
        # few fewer, but never more.
        assert 6 <= meta["total_frames"] <= 8
        assert meta["width"] == 32
        assert meta["height"] == 24
        assert meta["fps"] > 0
        assert meta["duration_s"] == pytest.approx(
            meta["total_frames"] / meta["fps"]
        )
    finally:
        handler.release()


def test_get_frame_is_rgb_not_bgr(make_test_video, tmp_path):
    """BGR→RGB regression: frame 3 was filled BGR (0,0,30); after cvtColor the
    QImage pixel is red (30,0,0). Forgetting cvtColor would make it blue."""
    path = make_test_video(tmp_path, frames=8)
    handler = VideoHandler(path)
    try:
        qimg = handler.get_frame(3)
        assert isinstance(qimg, QImage)
        assert not qimg.isNull()
        px = qimg.pixel(0, 0)
        r, g, b = qRed(px), qGreen(px), qBlue(px)
        assert r > g and r > b            # red channel dominant, not blue
        assert r == pytest.approx(30, abs=8)
    finally:
        handler.release()


def test_get_frame_out_of_range_returns_none(make_test_video, tmp_path):
    path = make_test_video(tmp_path, frames=8)
    handler = VideoHandler(path)
    try:
        assert handler.get_frame(-1) is None
        assert handler.get_frame(handler.total_frames) is None
        assert handler.get_frame(9999) is None
    finally:
        handler.release()


def test_release_is_idempotent(make_test_video, tmp_path):
    path = make_test_video(tmp_path, frames=8)
    handler = VideoHandler(path)
    handler.release()
    handler.release()  # must not raise
    # A released handler decodes nothing.
    assert handler.get_frame(0) is None


def test_constructor_raises_on_bad_path(tmp_path):
    bad = tmp_path / "not_a_video.avi"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        VideoHandler(str(bad))


# ── VideoSliceProvider + LazySliceList ───────────────────────────────────────

def test_provider_names_and_extract(make_test_video, tmp_path):
    path = make_test_video(tmp_path, frames=8)
    handler = VideoHandler(path)
    try:
        provider = VideoSliceProvider(handler, "clip")
        assert provider.names == [
            frame_key("clip", i) for i in range(handler.total_frames)
        ]
        img = provider.extract(frame_key("clip", 2))
        assert isinstance(img, QImage) and not img.isNull()
        # An unknown / non-frame name never decodes.
        assert provider.extract("clip_T1_Z2") is None
    finally:
        handler.release()


def test_lazy_slice_list_decodes_on_demand_through_lru(make_test_video, tmp_path):
    """A LazySliceList over a VideoSliceProvider decodes each frame only when
    asked and caches it in the shared SliceLRU (issue #45 reconciliation)."""
    path = make_test_video(tmp_path, frames=8)
    handler = VideoHandler(path)
    try:
        # Spy on the actual decode calls.
        calls = {"n": 0}
        original = handler.get_frame

        def counting(idx):
            calls["n"] += 1
            return original(idx)

        handler.get_frame = counting

        provider = VideoSliceProvider(handler, "clip")
        lazy = LazySliceList(provider)
        pid = lazy.provider_id
        lru = get_shared_lru()

        assert len(lazy) == handler.total_frames
        # Constructing the list decodes nothing.
        assert calls["n"] == 0
        assert lru.count_prefix(pid) == 0

        first = lazy.get(lazy.names[0])
        assert isinstance(first, QImage)
        assert calls["n"] == 1
        assert lru.count_prefix(pid) == 1

        # A second get of the SAME frame is an LRU hit — no new decode.
        lazy.get(lazy.names[0])
        assert calls["n"] == 1

        # A different frame decodes once more and caches.
        lazy.get(lazy.names[2])
        assert calls["n"] == 2
        assert lru.count_prefix(pid) == 2

        # release() drops this provider's cached frames from the shared LRU.
        lazy.release()
        assert lru.count_prefix(pid) == 0
    finally:
        handler.release()
