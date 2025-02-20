"""
Utility functions for the Image Annotator application.

This module contains helper functions used across the application.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import numpy as np

def calculate_area(annotation):
    if "segmentation" in annotation:
        # Polygon area
        x, y = annotation["segmentation"][0::2], annotation["segmentation"][1::2]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    elif "bbox" in annotation:
        # Rectangle area
        x, y, w, h = annotation["bbox"]
        return w * h
    return 0

def calculate_bbox(segmentation):
    x_coordinates, y_coordinates = segmentation[0::2], segmentation[1::2]
    x_min, y_min = min(x_coordinates), min(y_coordinates)
    x_max, y_max = max(x_coordinates), max(y_coordinates)
    width, height = x_max - x_min, y_max - y_min
    return [x_min, y_min, width, height]

def normalize_image(image_array):
    """Normalize image array to 8-bit range."""
    if image_array.dtype != np.uint8:
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    return image_array

