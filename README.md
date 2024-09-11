# DigitalSreeni Image Annotator

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI version](https://img.shields.io/pypi/v/digitalsreeni-image-annotator.svg?style=flat-square)

A powerful and user-friendly tool for annotating images with polygons and rectangles, built with PyQt5.

![DigitalSreeni Image Annotator Demo](screenshots/digitalsreeni-image-annotator-demo.gif)


@DigitalSreeni
Dr. Sreenivas Bhattiprolu

## Features

- Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?
- Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge.
- Save annotations in COCO-compatible JSON format.
- Edit existing annotations.
- Load and continue previous annotation work.
- Handle multi-dimensional images (TIFF stacks and CZI files).
- Zoom and pan for detailed annotations.
- Support for multiple classes with customizable colors.
- Import and export annotations.
- User-friendly interface with intuitive controls.
- Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires.
- Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?
- NEW: Load custom SAM2 pre-trained models for flexible and improved semi-automated annotations.

## Installation

You can install the DigitalSreeni Image Annotator directly from PyPI:

```bash
pip install digitalsreeni-image-annotator
```

### Important: PyQt5 Requirement

This application requires PyQt5 version 5.15.7 or higher. If you encounter any issues related to PyQt5, you may need to install or upgrade it separately:

```bash
pip install PyQt5>=5.15.7
```

If you're using an older version of PyQt5, please upgrade to ensure compatibility:

```bash
pip install --upgrade PyQt5>=5.15.7
```

On some systems, especially Linux, you might need to install additional system packages. For example, on Ubuntu or Debian:

```bash
sudo apt-get install python3-pyqt5
```

Note that the system package manager might not always provide the latest version. In such cases, using pip as shown above is recommended.

For other operating systems or if you encounter any issues, please refer to the [PyQt5 documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/installation.html).

### SAM-2 Installation for Semi-Automated Annotations

To use the Segment Anything Model (SAM-2) assisted annotations feature, you need to install PyTorch and SAM-2 libraries. Follow these steps:

1. Install PyTorch:
   - For CPU-only installation: `pip install torch torchvision`
   - For full installation instructions, visit: https://pytorch.org/get-started/locally/

2. Install SAM-2:
   - Visit the SAM-2 repository: https://github.com/facebookresearch/segment-anything-2
   - Follow these key steps:
     a. Open a console (command prompt). We recommend using Anaconda Prompt for easy environment management.
     b. Ensure you're in the right environment. If needed, create a new environment with the specified Python version and activate it.
     c. Change directory to where you want to download the repository (e.g., your Downloads folder).
     d. Clone the SAM-2 repository:
        ```bash
        git clone https://github.com/facebookresearch/segment-anything-2.git
        ```
        (This assumes you have Git installed. On Windows, you can use Git for Windows: https://gitforwindows.org/)
     e. Change to the repository directory:
        ```bash
        cd segment-anything-2
        ```
     f. Install SAM-2:
        ```bash
        pip install .
        ```
        (Note: Do not use `pip install -e .` if you intend to delete the downloaded repository after installation)

Your system is now ready to use SAM-2 for semi-automated annotations.

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
   - Click "Open New Image Set" to load a new set of images from your computer.
   - Use "Add More Images" to append images to the current set.
   - Add classes using the "Add Class" button.
   - Select a class and use the Polygon or Rectangle tool to create manual annotations.
   - To use SAM2 Magic Wand:
     - Click "Load SAM2 Model" and select the config (.yaml) and model (.pt) files.
     - Select the SAM2 Magic Wand button and draw a rectangle around your object of interest for automated annotation.
   - Edit existing annotations by double-clicking on them.
   - Save your annotations using the "Save Annotations" button.
   - Use "Import Saved Annotations" to load previously created annotations.
   - Access the help documentation by clicking the "Help" button.
   - Explore the interface – you might stumble upon some hidden gems and secret features!

3. Keyboard shortcuts:
   - Use the mouse wheel or trackpad to zoom in/out
   - Hold Ctrl and drag to pan the image
   - Press 'Esc' to cancel the current annotation
   - Press 'Enter' to finish the current polygon annotation or accept a SAM2-generated mask

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
