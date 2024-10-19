# DigitalSreeni Image Annotator and Toolkit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI version](https://img.shields.io/pypi/v/digitalsreeni-image-annotator.svg?style=flat-square)

A powerful and user-friendly tool for annotating images with polygons and rectangles, built with PyQt5. Now with additional supporting tools for comprehensive image processing and dataset management.

## Support the Project

If you find this project helpful, consider supporting it:

[![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/donate/?business=FGQL3CNJGJP9C&no_recurring=0&item_name=If+you+find+this+Image+Annotator+project+helpful%2C+consider+supporting+it%3A&currency_code=USD)

![DigitalSreeni Image Annotator Demo](screenshots/digitalsreeni-image-annotator-demo.gif)

## Watch the demo (of v0.8.0):
[![Watch the demo video](https://img.youtube.com/vi/aArn1f1YIQk/maxresdefault.jpg)](https://youtu.be/aArn1f1YIQk)

@DigitalSreeni
Dr. Sreenivas Bhattiprolu

## Features

- Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?
- Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge.
- Paint brush and Eraser tools with adjustable pen sizes (use - and = on your keyboard)
- Merge annotations - For when SAM-2's guesswork needs a little human touch. 
- Save and load projects for continued work.
- Save As... and Autosave functionality. 
- A secret game, for when you are bored.
- Import existing COCO JSON annotations with images.
- Export annotations to various formats (COCO JSON, YOLO v8/v11, Labeled images, Semantic labels, Pascal VOC).
- Handle multi-dimensional images (TIFF stacks and CZI files).
- Zoom and pan for detailed annotations.
- Support for multiple classes with customizable colors.
- User-friendly interface with intuitive controls.
- Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires.
- Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?
- Pick appropriate pre-trained SAM2 model for flexible and improved semi-automated annotations.
- Change the class of an annotation to a different class.
- Turn visibility of a class ON and OFF.
- YOLO (beta) training using current annotations and loading trained model to segment images.
- Area measurements for annotations displayed next to the Annotation name.
- Sort annotations by name/number or area.
- Additional supporting tools:
  - Annotation statistics for current annotations
  - COCO JSON combiner
  - Dataset splitter
  - Stack to slices converter
  - Image patcher
  - Image augmenter
- NEW: Project Details: View and edit project metadata, including creation date, last modified date, image information, and custom notes.
- NEW: Advanced Project Search: Search through multiple projects using complex queries with logical operators (AND, OR) and parentheses.


## Operating System Requirements
This application is built using PyQt5 and has been tested on macOS and Windows. It may experience compatibility issues on Linux systems, particularly related to the XCB plugin for PyQt5. Extensive testing on Linux systems has not been done yet.

## Installation

### Watch the installation walkthough video:
[![Watch the installation video](https://img.youtube.com/vi/VI6V95eUUpY/maxresdefault.jpg)](https://youtu.be/VI6V95eUUpY)

You can install the DigitalSreeni Image Annotator directly from PyPI:

```bash
pip install digitalsreeni-image-annotator
```

The application uses the Ultralytics library, so there's no need to separately install SAM2 or PyTorch, or download SAM2 models manually.

## Usage

1. Run the DigitalSreeni Image Annotator application:
   ```bash
   digitalsreeni-image-annotator
   ```
   or
    ```bash
    sreeni
    ```
   or
   ```bash
   python -m digitalsreeni_image_annotator.main
   ```

2. Using the application:
   - Click "New Project" or use Ctrl+N to start a new project.
   - Use "Add New Images" to import images, including TIFF stacks and CZI files.
   - Add classes using the "Add Classes" button.
   - Select a class and use the Polygon or Rectangle or Paint Brush tool to create manual annotations.
   - To use SAM2-assisted annotation:
     - Select a model from the "Pick a SAM Model" dropdown. It's recommended to use smaller models like SAM2 tiny or SAM2 small. SAM2 large is not recommended as it may crash the application on systems with limited resources.  
     - Note: When you select a model for the first time, the application needs to download it. This process may take a few seconds to a minute, depending on your internet connection speed. Subsequent uses of the same model will be faster as it will already be cached locally, in your working directory.
     - Click the "SAM-Assisted" button to activate the tool.
     - Draw a rectangle around objects of interest to allow SAM2 to automatically detect objects.
     - Note that SAM2 provides various outputs with different scores, and only the top-scoring region will be displayed. If the desired result isn't achieved on the first try, draw again.
     - For low-quality images where SAM2 may not auto-detect objects, manual tools may be necessary.
     - When SAM2 auto-detect partial objects, use polygon or paint brush tools to manually define the remaining region and use the Merge tool to combine both annotations into one.
     - When SAM2 over-annotates objects, extending the annotation beyond object's boundaries, use the Eraser tool to clean up the edges. 
     - Both paint brush and eraser tools can be adjusted for pen size by using - or = keys on your keyboard.  
   - Edit existing annotations by double-clicking on them.
   - Edit existing annotations using the Eraser tool. Adjust the eraser size by using - or = keys on your keyboard.
   - Merge connected annotations by selecting them from the Annotations list and clicking the Merge button. 
   - Change the class of an annotation to a different class.
   - Turn visibility of a class ON and OFF.
   - Use YOLO (beta) training with current annotations and load the trained model to segment images and convert segmentations to annotations. (Currently not implemented for slices or stacks, just single images.)
   - Accept/reject one or select class predictions at a time to add them as annotations.
   - View area measurements for annotations displayed next to the Annotation name.
   - Sort annotations by name/number or area.
   - Save your project using "Save Project" or Ctrl+S. Alternatively, you can use Save As... to save the project with a different name. 
   - Use "Open Project" or Ctrl+O to load a previously saved project.
   - Click "Import Annotations with Images" to load existing COCO JSON annotations along with their images.
   - Use "Export Annotations" to save annotations in various formats (COCO JSON, YOLO v8/v11, Labeled images, Semantic labels, Pascal VOC).
     - Note: YOLO export (and import) is now compatible with YOLOv11 structure. (Project directory includes data.yaml, train, and valid directories, with train and valid both having images and labels subdirectories.)
   - Project Details:
     - Access project details by selecting "Project Details" from the Project menu.
     - View project metadata such as creation date, last modified date, and image information.
     - Add or edit custom project notes.
     - Project details are automatically saved when you make changes to the notes.
   - Advanced Project Search:
     - Access the search functionality by selecting "Search Projects" from the Project menu.
     - Search through multiple projects using complex queries.
     - Use logical operators (AND, OR) and parentheses for advanced search criteria.
     - Search covers project name, class names, image names, and project notes.
     - Example queries:
       - "cells AND dog": Find projects containing both "cells" and "dog"
       - "cells OR bacteria": Find projects containing either "cells" or "bacteria"
       - "cells AND (dog OR monkey)": Find projects containing "cells" and either "dog" or "monkey"
       - "(project1 OR project2) AND (cells OR bacteria)": More complex nested queries
     - Double-click on search results to open the corresponding project.
   - Access additional tools under the Tools menu bar:
     - Annotation Statistics
     - COCO JSON Combiner
     - Dataset Splitter
     - Stack to Slices Converter
     - Image Patcher
     - Image Augmenter
   - Each tool opens a separate UI to guide you through the respective task.
   - Access the help documentation by clicking the "Help" button or pressing F1.
   - Explore the interface – you might stumble upon some hidden gems and secret features!

3. Keyboard shortcuts:
   - Ctrl + N: Create a new project
   - Ctrl + O: Open an existing project
   - Ctrl + S: Save the current project
   - Ctrl + W: Close the current project
   - Ctrl + Shift + S: Open Annotation Statistics
   - F1: Open the help window
   - Ctrl + Wheel: Zoom in/out
   - Hold Ctrl and drag: Pan the image
   - Esc: Cancel current annotation, exit edit mode, or exit SAM-assisted annotation
   - Enter: Finish current annotation, exit edit mode, or accept SAM-generated mask
   - Up/Down Arrow Keys: Navigate through slices in multi-dimensional images
   - - and =: Adjust pen size for paint brush and eraser tools

## Known Issues and Bug Fixes

- The application may not work correctly on Linux systems. Extensive testing has not been done yet.
- When loading a YOLO model trained on different classes compared to the loaded YAML file, the application now gives a message to the user about the mismatch instead of crashing.
- Various other bugs have been addressed to improve overall stability and performance.

## Development

For development purposes, you can clone the repository and install it in editable mode:

1. Clone the repository:
   ```bash
   git clone https://github.com/bnsreenu/digitalsreeni-image-annotator.git
   cd digitalsreeni-image-annotator
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package and its dependencies in editable mode:
   ```bash
   pip install -e .
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all my [YouTube](http://www.youtube.com/c/DigitalSreeni) subscribers who inspired me to work on this project
- Inspired by the need for efficient image annotation in computer vision tasks

## Contact

Dr. Sreenivas Bhattiprolu - [@DigitalSreeni](https://twitter.com/DigitalSreeni)

Project Link: [https://github.com/bnsreenu/digitalsreeni-image-annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)

## Citing

If you use this software in your research, please cite it as follows:

Bhattiprolu, S. (2024). DigitalSreeni Image Annotator [Computer software]. 
https://github.com/bnsreenu/digitalsreeni-image-annotator

```bibtex
@software{digitalsreeni_image_annotator,
  author = {Bhattiprolu, Sreenivas},
  title = {DigitalSreeni Image Annotator},
  year = {2024},
  url = {https://github.com/bnsreenu/digitalsreeni-image-annotator}
}
```
