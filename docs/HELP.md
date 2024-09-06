# Image Annotator Help Guide

## Overview

Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in COCO-style JSON format, including both polygon (segmentation) and bounding box information. The tool now supports multi-dimensional images such as TIFF stacks and CZI files.

## Key Features

- Easy-to-use graphical interface
- Support for polygon and rectangle annotations
- COCO-style JSON output with segmentation and bounding box data
- Ability to load and continue previous annotation work
- Support for multiple image formats (png, jpg, bmp, tif, tiff, czi)
- Multi-class annotation support with customizable colors
- Handling of multi-dimensional images (TIFF stacks and CZI files)

## Getting Started

### Starting a New Project

1. Click "Open New Image Set" to import multiple images you want to annotate, including TIFF stacks and CZI files.
2. For multi-dimensional images, you'll be prompted to assign dimensions (e.g., T for time, Z for depth, C for channels).
3. Use "Add Class" to define classes of interest.
4. Start annotating by selecting a class and using the Polygon or Rectangle Tool.

### Continuing a Previous Project

1. Click "Import Saved Annotations" to load your previous work.
2. Use "Open New Image Set" to load the corresponding images.
3. If needed, use "Add More Images" to include additional images for annotation.

## Annotation Process

1. **Select a Class**: Choose the class you want to annotate from the class list.
2. **Choose a Tool**: Select either the Polygon Tool (recommended for better control) or Rectangle Tool.
3. **Create Annotation**: 
   - For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.
   - For Rectangle Tool: Click and drag to create a bounding box.
4. **Finish Annotation**: Hit Enter or click "Finish Polygon" to complete the annotation.

## Navigation and Viewing

- **Zoom**: Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.
- **Pan**: Hold Ctrl, click the left mouse button, and move the mouse.
- **Switch Images**: Click on an image name in the image list on the right.
- **Navigate Slices**: For multi-dimensional images, use the slice list on the right.

## Handling Multi-dimensional Images

- When opening a TIFF stack or CZI file, you'll be prompted to assign dimensions (e.g., T, Z, C, H, W).
- The slice list on the right will show all available slices for the current image.
- Annotations are specific to each slice and will be saved accordingly.
- Use the up/down arrow keys to navigate through slices quickly.
- Slices with annotations will be highlighted in green in the slice list.

## Editing Annotations

1. Ensure you're not in annotation mode (tool buttons should be grey, not blue).
2. Double-click an existing annotation to enter edit mode.
3. Modify the annotation:
   - Move points: Click and drag existing points.
   - Add points: Click on the boundary line.
   - Delete points: Shift + click on existing points.
4. Press Enter to accept the edits.

## Managing Classes

- **Add Class**: Click the "Add Class" button.
- **Change Class Color**: Right-click a class, select "Change Color", and choose from the palette.
- **Rename Class**: Right-click a class and select "Rename Class".
- **Delete Class**: Right-click a class and select "Delete Class".

## Managing Annotations

- **View Annotations**: Check the Annotations list at the bottom left of the GUI.
- **Highlight Annotations**: Select an annotation from the list to highlight it in red on the image.
- **Multi-select Annotations**: Hold Ctrl and click multiple annotations.
- **Delete Annotations**: Select annotation(s) and click "Delete Selected Annotations".

## Saving Your Work

- Click "Save Annotations" to save your work as a COCO-style JSON file.
- For multi-dimensional images, annotated slices will be saved as separate PNG files along with the JSON file.
- You can close the program after saving and continue your work later by importing the saved annotations.

## Supported Formats

- Supports common image formats: PNG, JPG, BMP
- Now supports multi-dimensional formats: TIFF (.tif, .tiff) and CZI (.czi)

## Known Issues

- When working with multiple images of different types (e.g., single images and image stacks), ensure you've selected the correct image before saving annotations to avoid any potential issues with slice saving.

## Getting Help

If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.