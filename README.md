# DigitalSreeni Image Annotator and Toolkit

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI version](https://img.shields.io/pypi/v/digitalsreeni-image-annotator.svg?style=flat-square)

A powerful and user-friendly tool for annotating images with polygons and rectangles, built with PyQt5. Now with additional supporting tools for comprehensive image processing and dataset management.

## Support the Project

If you find this project helpful, consider supporting it:

[![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/donate/?business=FGQL3CNJGJP9C&no_recurring=0&item_name=If+you+find+this+Image+Annotator+project+helpful%2C+consider+supporting+it%3A&currency_code=USD)



![DigitalSreeni Image Annotator Demo](screenshots/digitalsreeni-image-annotator-demo.gif)

## Watch the demo:
[![Watch the demo video](https://img.youtube.com/vi/BupyYUw2boI/maxresdefault.jpg)](https://youtu.be/BupyYUw2boI)


@DigitalSreeni
Dr. Sreenivas Bhattiprolu

## Features

- Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?
- Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge.
- Merge annotations - For when SAM-2's guesswork needs a little human touch. 
- Save and load projects for continued work.
- Import existing COCO JSON annotations with images.
- Export annotations to various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC).
- Handle multi-dimensional images (TIFF stacks and CZI files).
- Zoom and pan for detailed annotations.
- Support for multiple classes with customizable colors.
- User-friendly interface with intuitive controls.
- Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires.
- Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?
- Pick appropriate pre-trained SAM2 model for flexible and improved semi-automated annotations.
- Additional supporting tools:
  - Annotation statistics for current annotations
  - COCO JSON combiner
  - Dataset splitter
  - Stack to slices converter
  - Image patcher
  - Image augmenter

## Operating System Requirements
This application is built using PyQt5 and has been tested on macOS and Windows. It may experience compatibility issues on Linux systems, particularly related to the XCB plugin for PyQt5.

## Installation

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
   - Select a class and use the Polygon or Rectangle tool to create manual annotations.
   - To use SAM2-assisted annotation:
     - Select a model from the "Pick a SAM Model" dropdown. It's recommended to use smaller models like SAM2 tiny or SAM2 small. SAM2 large is not recommended as it may crash the application on systems with limited resources.  
     - Note: When you select a model for the first time, the application needs to download it. This process may take a few seconds to a minute, depending on your internet connection speed. Subsequent uses of the same model will be faster as it will already be cached locally, in your working directory.
     - Click the "SAM-Assisted" button to activate the tool.
     - Draw a rectangle around objects of interest to allow SAM2 to automatically detect objects.
     - Note that SAM2 provides various outputs with different scores, and only the top-scoring region will be displayed. If the desired result isn't achieved on the first try, draw again.
     - For low-quality images where SAM2 may not auto-detect objects, manual tools may be necessary.
   - Edit existing annotations by double-clicking on them.
   - Merge connected annotations by selecting them from the Annotations list and clicking the Merge button. 
   - Save your project using "Save Project" or Ctrl+S.
   - Use "Open Project" or Ctrl+O to load a previously saved project.
   - Click "Import Annotations with Images" to load existing COCO JSON annotations along with their images.
   - Use "Export Annotations" to save annotations in various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC).
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
