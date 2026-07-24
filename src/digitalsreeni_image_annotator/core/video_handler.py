"""Video loading with frames exposed as lazy "slices" (issue #47).

A video is treated as a multi-dimensional stack whose slices are its frames.
Rather than inventing a parallel frame cache, this module plugs into the
issue-#45 lazy-slice machinery (:mod:`core.slice_cache`): a
:class:`VideoSliceProvider` is duck-type compatible with
``slice_cache.SliceProvider`` (it exposes ``provider_id``, ``names`` and
``extract(name)``), so a video's ``image_slices[base]`` is an ordinary
:class:`~core.slice_cache.LazySliceList`. Every existing slice consumer
(``switch_slice``, ``activate_slice``, exporters, DINO batch, save, delete)
therefore works for video unchanged, and the shared bounded ``SliceLRU``
caps how many decoded frame QImages are held live at once.

Frame decoding uses OpenCV (``cv2.VideoCapture``). ``cv2.VideoCapture`` is
**not thread-safe**, so ``VideoHandler`` must be driven from the GUI thread
only. It performs *decoding only* — no internal LRU; the shared SliceLRU
does the caching (a fresh QImage per decode keeps the SAM worker
ADR-013-safe).

This module deliberately imports no main-window code, so it is unit-testable
headless (only ``cv2`` and ``QImage`` are needed).
"""

import re

import cv2
from PyQt6.QtGui import QImage

# Recognised video container extensions (case-insensitive).
VIDEO_EXTS = (".mp4", ".avi", ".mov")

# Frame slice keys look like "<base>_F00042" — 0-based frame index, zero
# padded to 5 digits, anchored at the END of the name so it never matches a
# multi-dim slice key like "stack_T1_Z5".
FRAME_KEY_RE = re.compile(r"_F(\d+)$")


def is_video(file_name):
    """True if ``file_name`` has a recognised video extension (#47)."""
    return file_name.lower().endswith(VIDEO_EXTS)


# Image container globs the app can open. Paired with VIDEO_EXTS to build the
# file-open dialog filter. Kept next to VIDEO_EXTS so the dialog and is_video()
# can never drift on the video part -- the #47 gap was exactly three filter
# string literals updated for video and one (add_images) missed. (#47/#48)
_IMAGE_GLOBS = "*.png *.jpg *.bmp *.tif *.tiff *.czi"


def file_dialog_filter():
    """Qt ``getOpenFileNames`` filter string for images + videos.

    The video globs are derived from :data:`VIDEO_EXTS`, so registering a new
    container there updates every Add/Open dialog at once instead of the app
    silently accepting a format no file picker ever offers.
    """
    video_globs = " ".join(f"*{ext}" for ext in VIDEO_EXTS)
    return f"Images & Videos ({_IMAGE_GLOBS} {video_globs})"


def frame_key(base_name, idx):
    """Slice name for frame ``idx`` (0-based) of the video ``base_name``."""
    return f"{base_name}_F{idx:05d}"


def parse_frame_index(slice_name):
    """Frame index encoded in ``slice_name``, or ``None``.

    Matches ``_F<digits>`` anchored at the end of the name so ordinary
    multi-dim slice keys (``stack_T1_Z5``) return ``None``.
    """
    match = FRAME_KEY_RE.search(slice_name)
    return int(match.group(1)) if match else None


class VideoHandler:
    """Decode individual frames of a video on demand (GUI-thread only).

    Metadata is read once at construction; ``get_frame`` seeks and decodes a
    single frame per call and always returns a fresh, self-owned QImage.
    """

    def __init__(self, path):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self._released = False

        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 (or NaN) fps; fall back to a sane default
        # so duration_s and the info line stay finite.
        self.fps = fps if fps and fps > 0 else 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = self.total_frames / self.fps if self.fps else 0.0

    def get_frame(self, idx):
        """Return frame ``idx`` as a QImage, or ``None``.

        Out-of-range indices, a failed decode, or a released handler all
        yield ``None``. OpenCV decodes BGR; we convert to RGB before building
        the QImage. The ``.copy()`` is MANDATORY — the numpy buffer backing
        the QImage is freed when this method returns, so the QImage must own
        its pixels.
        """
        if self._released or idx is None or not (0 <= idx < self.total_frames):
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888
        )
        return qimage.copy()

    def metadata(self):
        """Return the video's metadata as a plain, JSON-serializable dict."""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
        }

    def release(self):
        """Release the capture. Idempotent."""
        if not self._released:
            self._cap.release()
            self._released = True


class VideoSliceProvider:
    """Duck-type compatible with :class:`core.slice_cache.SliceProvider`.

    Exposes ``provider_id`` / ``names`` / ``extract(name)`` so a
    :class:`~core.slice_cache.LazySliceList` can wrap it and the shared
    SliceLRU caches the decoded frames (a fresh QImage per decode).
    """

    def __init__(self, video_handler, base_name):
        self._handler = video_handler
        self.base_name = base_name
        self.provider_id = id(self)
        self.names = [
            frame_key(base_name, i) for i in range(video_handler.total_frames)
        ]

    def extract(self, name):
        """Decode the frame named ``name`` to a QImage (``None`` if unknown)."""
        idx = parse_frame_index(name)
        if idx is None:
            return None
        return self._handler.get_frame(idx)
