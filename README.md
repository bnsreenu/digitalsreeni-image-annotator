# Image Annotator

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A powerful and user-friendly tool for annotating images with polygons and rectangles, built with PyQt5.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu

## Features

- Load and annotate images with polygons and rectangles
- Save annotations in COCO-compatible JSON format
- Edit existing annotations
- Zoom and pan functionality for detailed annotations
- Support for multiple classes with customizable colors
- Import and export annotations
- User-friendly interface with intuitive controls

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/image-annotator.git
   cd image-annotator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package and its dependencies:
   ```
   pip install -e .
   ```

## Usage

1. Run the Image Annotator application:
   ```
   python -m image_annotator.main
   ```

2. Using the application:
   - Click "Open New Image Set" to load a new set of images from your computer.
   - Use "Add More Images" to append images to the current set.
   - Add classes using the "Add Class" button.
   - Select a class and use the Polygon or Rectangle tool to create annotations.
   - Edit existing annotations by double-clicking on them.
   - Save your annotations using the "Save Annotations" button.
   - Use "Import Saved Annotations" to load previously created annotations.

3. Keyboard shortcuts:
   - Use the mouse wheel or trackpad to zoom in/out
   - Hold Ctrl and drag to pan the image
   - Press 'Esc' to cancel the current annotation
   - Press 'Enter' to finish the current polygon annotation

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

- Thanks to all contributors who have helped shape this project
- Inspired by the need for efficient image annotation in computer vision tasks

## Contact

Dr. Sreenivas Bhattiprolu - [@DigitalSreeni](https://twitter.com/DigitalSreeni)

Project Link: [https://github.com/bnsreenu/image_annotator](https://github.com/bnsreenu/image_annotator)

## Citing

If you use this software in your research, please cite it as follows:

Bhattiprolu, S. (2024). Image Annotator [Computer software]. 
https://github.com/bnsreenu/image_annotator

```
@software{image_annotator,
  author = {Bhattiprolu, Sreenivas},
  title = {Image Annotator},
  year = {2024},
  url = {https://github.com/bnsreenu/image_annotator}
}
```