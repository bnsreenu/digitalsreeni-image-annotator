"""
Image Annotator
===============
A tool for annotating images with polygons and rectangles.
This package provides a GUI application for image annotation,
supporting polygon and rectangle annotations in a COCO-compatible format.
@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""
__version__ = "0.9.1"
__author__ = "Dr. Sreenivas Bhattiprolu"

# Lazy loading — importing this package must NOT pull in PyQt6, because
# main.py needs to import torch BEFORE Qt loads (ADR-017).  The modules
# below transitively import PyQt6.  Deferring them to __getattr__ keeps
# ``import digitalsreeni_image_annotator`` cheap and Qt-free.
__all__ = [
    "ImageAnnotator",
    "ImageLabel",
    "calculate_area",
    "calculate_bbox",
    "SAMUtils",
]


def __getattr__(name):
    if name == "ImageAnnotator":
        from .annotator_window import ImageAnnotator
        return ImageAnnotator
    if name == "ImageLabel":
        from .widgets.image_label import ImageLabel
        return ImageLabel
    if name == "SAMUtils":
        from .inference.sam_utils import SAMUtils
        return SAMUtils
    if name == "calculate_area":
        from .utils import calculate_area
        return calculate_area
    if name == "calculate_bbox":
        from .utils import calculate_bbox
        return calculate_bbox
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
