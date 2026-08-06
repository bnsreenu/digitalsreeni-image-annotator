"""Read an image's pixel dimensions without Qt (issue #76).

The export layer needed one thing from ``QImage``: how big a file is. That one
need was enough to make ``io/export_formats.py`` import PyQt6 at module level,
and therefore enough to make a headless export require a display.

Pillow is already a dependency and reads the header without decoding the
pixels, which is both Qt-free and strictly faster than constructing a QImage
just to ask for its width.
"""

from .logging_config import get_logger

logger = get_logger(__name__)


def image_dimensions(path: str) -> tuple[int, int]:
    """``(width, height)`` of the image at ``path``, or ``(0, 0)``.

    ``(0, 0)`` rather than an exception on an unreadable file: this mirrors
    ``QImage(path)``, which yields a null image with zero size, so the export
    paths that previously relied on that behaviour keep working unchanged. The
    callers already guard against zero dimensions.
    """
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Image.DecompressionBombError:
        # Pillow refuses images above ~178 MP by default, where QImage would
        # simply have read the header. Raise the guard for this one call rather
        # than silently reporting a giant slide as 0x0 — we are reading a
        # header, not decoding pixels, so the bomb protection buys nothing here.
        previous, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
        try:
            with Image.open(path) as image:
                return image.width, image.height
        except Exception:
            logger.warning("could not read image dimensions from %s", path)
            return 0, 0
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    except Exception:
        logger.warning("could not read image dimensions from %s", path)
        return 0, 0
