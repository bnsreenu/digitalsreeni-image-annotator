"""
Utility functions for the Image Annotator application.

This module contains helper functions used across the application.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

def calculate_area(annotation):
    """
    Calculate the area of an annotation.

    Args:
        annotation (dict): The annotation dictionary containing either
                           'segmentation' or 'bbox' key.

    Returns:
        float: The calculated area.
    """
    if "segmentation" in annotation:
        x, y = annotation["segmentation"][0::2], annotation["segmentation"][1::2]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    elif "bbox" in annotation:
        x, y, width, height = annotation["bbox"]
        return width * height

def calculate_bbox(segmentation):
    """
    Calculate the bounding box from a segmentation.

    Args:
        segmentation (list): List of x and y coordinates of the segmentation.

    Returns:
        list: Bounding box in the format [x_min, y_min, width, height].
    """
    x_coordinates = segmentation[0::2]
    y_coordinates = segmentation[1::2]
    x_min = min(x_coordinates)
    y_min = min(y_coordinates)
    x_max = max(x_coordinates)
    y_max = max(y_coordinates)
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

