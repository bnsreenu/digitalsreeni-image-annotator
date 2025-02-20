"""
Image Annotator
===============
A tool for annotating images with polygons and rectangles.
This package provides a GUI application for image annotation,
supporting polygon and rectangle annotations in a COCO-compatible format.
@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""
__version__ = "0.8.12"
__author__ = "Dr. Sreenivas Bhattiprolu"

from .annotator_window import ImageAnnotator
from .image_label import ImageLabel
from .utils import calculate_area, calculate_bbox
from .sam_utils import SAMUtils  

__all__ = ['ImageAnnotator', 'ImageLabel', 'calculate_area', 'calculate_bbox', 'SAMUtils']  # Add 'SAMUtils' to this list