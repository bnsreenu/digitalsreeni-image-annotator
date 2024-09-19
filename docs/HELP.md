# Image Annotator Help Guide

## Overview

Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in various formats, including COCO-style JSON, YOLO v8, and Pascal VOC. Annotations can be defined using manual tools like the polygon tool or in a semi-automated way with the assistance of the Segment Anything Model (SAM-2) pre-trained model. The tool supports multi-dimensional images such as TIFF stacks and CZI files and provides dark mode and adjustable application font sizes for enhanced GUI experience.

## Key Features

- Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?
- Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge
- Save and load projects for continued work
- Import existing COCO JSON annotations with images
- Export annotations to various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC)
- Handle multi-dimensional images (TIFF stacks and CZI files)
- Zoom and pan for detailed annotations
- Support for multiple classes with customizable colors
- User-friendly interface with intuitive controls
- Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires
- Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?
- Support for common image formats (PNG, JPG, BMP) and multi-dimensional formats (TIFF, CZI)
- Load custom SAM2 pre-trained models for flexible and improved semi-automated annotations

## Getting Started

### Starting a New Project

1. Click "New Project" or use Ctrl+N to start a new project.
2. Click "Add New Images" to import multiple images you want to annotate, including TIFF stacks and CZI files.
3. For multi-dimensional images, you'll be prompted to assign dimensions (e.g., T for time, Z for depth, C for channels).
4. Use "Add Classes" to define classes of interest.
5. Start annotating by selecting a class and using the Polygon, Rectangle Tool, or SAM2 Magic Wand.

### Opening an Existing Project

1. Click "Open Project" or use Ctrl+O to load a previously saved project.
2. If there are any missing images, you'll be prompted to locate them on your drive. Located images will be automatically copied to the project directory.
3. If you choose not to locate missing images, the annotations for those images will be removed.

### Importing Existing Annotations

1. Click "Import Annotations with Images" to load existing COCO JSON annotations along with their corresponding images.
2. Select the COCO JSON file. The images should be in the same directory as the JSON file.
3. The annotations and images will be loaded into your current project.

## Loading SAM2 Pre-trained Models

1. Download the desired SAM2 model file (.pt) and its corresponding config file (.yaml) from the [Segment Anything 2 GitHub repository](https://github.com/facebookresearch/segment-anything-2).
2. For the model file, navigate to the main repository page and download the desired .pt file.
3. For the config file, go to the [sam2_configs folder](https://github.com/facebookresearch/segment-anything-2/tree/main/sam2_configs) and download the corresponding .yaml file.
4. Ensure you download matching pairs, for example:
   - For sam2_hiera_small.pt model, download sam2_hiera_s.yaml config file
   - For sam2_hiera_tiny.pt model, download sam2_hiera_t.yaml config file
5. In the Image Annotator, click the "Load SAM2 Model" button in the Automated Tools section.
6. First, select the config file (.yaml) when prompted.
7. Then, select the corresponding model file (.pt) when prompted.
8. Once loaded successfully, the SAM2 Magic Wand button will become active for use.

## Annotation Process

1. **Select a Class:** Choose the class you want to annotate from the class list.
2. **Choose a Tool:** Select either the Polygon Tool, Rectangle Tool, or SAM2 Magic Wand.
3. **Create Annotation:**
   - For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.
   - For Rectangle Tool: Click and drag to create a bounding box.
   - For SAM2 Magic Wand: Ensure a SAM2 model is loaded. Click the SAM2 Magic Wand button to activate assisted annotation. Click and drag around an object, and SAM2 will display the segmented mask. Press Enter to accept the annotation, continue drawing to refine it, or press Escape to exit SAM-assisted annotation.

## Exporting Annotations

1. Click "Export Annotations" to open the export dialog.
2. Select the desired export format from the dropdown menu:
   - **COCO JSON:** Exports a JSON file in COCO format. Save it in the same directory as the images for easy reimport.
   - **YOLO v8:** Exports txt files for each image with annotations, along with a yaml file, saved in a 'labels' directory.
   - **Labeled images:** Saves labeled images for each class in separate directories.
   - **Semantic labels:** Exports semantic label images where each class is represented by a unique pixel value.
   - **Pascal VOC BBox:** Exports XML files with bounding box annotations in Pascal VOC format.
   - **Pascal VOC BBox + Segmentation:** Exports XML files with both bounding box and segmentation annotations in Pascal VOC format.
3. Choose the export location and confirm to save the annotations in the selected format.

## Navigation and Viewing

- **Zoom:** Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.
- **Pan:** Hold Ctrl, click the left mouse button, and move the mouse.
- **Switch Images:** Click on an image name in the image list on the right.
- **Navigate Slices:** For multi-dimensional images, use the slice list on the right or use the up/down arrow keys to move through slices.

## Handling Multi-dimensional Images

- When opening a TIFF stack or CZI file, you'll be prompted to assign dimensions (e.g., T, Z, S, C, H, W).
- The slice list on the right will show all available slices for the current image.
- Annotations are specific to each slice and will be saved accordingly.
- Slices with annotations will be highlighted in green in the slice list.
- You can reassign dimensions after import by right-clicking on the image name. Note that all annotations for that image will be lost when dimensions are changed after importing and annotating.

## Editing Annotations

1. Ensure you're not in annotation mode (tool buttons should be grey, not blue).
2. Double-click an existing annotation to enter edit mode.
3. Modify the annotation:
   - Move points: Click and drag existing points.
   - Add points: Click on the boundary line.
   - Delete points: Shift + click on existing points.
4. Press Enter to accept the edits.

## Managing Classes

- **Add Class:** Click the "Add Classes" button.
- **Change Class Color:** Right-click a class, select "Change Color", and choose from the palette.
- **Rename Class:** Right-click a class and select "Rename Class".
- **Delete Class:** Right-click a class and select "Delete Class".

## Managing Annotations

- **View Annotations:** Check the Annotations list at the bottom left of the GUI.
- **Highlight Annotations:** Select an annotation from the list to highlight it in red on the image.
- **Multi-select Annotations:** Hold Ctrl and click multiple annotations.
- **Delete Annotations:** Select annotation(s) and click "Delete Selected Annotations".

## Saving Your Work

- Click "Save Project" or use Ctrl+S to save your current project.
- You can close the program after saving and continue your work later by opening the saved project.

## Customization

- **Dark Mode:** Click the "Toggle Dark Mode" button to switch between light and dark themes for comfortable viewing in different lighting conditions.
- **Font Size:** Use the Font Size drop-down selector to adjust the application's font size to your comfort level. Make your annotations as big or small as your caffeine level requires!

## Keyboard Shortcuts

- **Ctrl + N:** Create a new project
- **Ctrl + O:** Open an existing project
- **Ctrl + S:** Save the current project
- **Ctrl + W:** Close the current project
- **F1:** Open the help window
- **Ctrl + Wheel:** Zoom in/out
- **Esc:** Cancel current annotation, exit edit mode, or exit SAM-assisted annotation
- **Enter:** Finish current annotation, exit edit mode, or accept SAM-generated mask
- **Up/Down Arrow Keys:** Navigate through slices in multi-dimensional images

## Known Issues

When opening images before loading saved annotations, the annotations may not display correctly for the first image. To avoid this issue, it is recommended to load saved annotations first, followed by opening the corresponding images.

## Getting Help

If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.
