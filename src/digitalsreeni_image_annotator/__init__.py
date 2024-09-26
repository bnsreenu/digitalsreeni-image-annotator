"""
Image Annotator
===============

A tool for annotating images with polygons and rectangles.

This package provides a GUI application for image annotation,
supporting polygon and rectangle annotations in a COCO-compatible format.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

__version__ = "0.5.8"
__author__ = "Dr. Sreenivas Bhattiprolu"

from .annotator_window import ImageAnnotator
from .image_label import ImageLabel
from .utils import calculate_area, calculate_bbox

__all__ = ['ImageAnnotator', 'ImageLabel', 'calculate_area', 'calculate_bbox']