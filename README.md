# DigitalSreeni Image Annotator

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI version](https://img.shields.io/pypi/v/digitalsreeni-image-annotator.svg?style=flat-square)

A powerful and user-friendly tool for annotating images using manual and automated tools, built with PyQt5.

![DigitalSreeni Image Annotator Demo](screenshots/digitalsreeni-image-annotator-demo.gif)


@DigitalSreeni
Dr. Sreenivas Bhattiprolu

## Features

- Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?
- Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge.
- Save and load projects for continued work.
- Import existing COCO JSON annotations with images.
- Export annotations to various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC).
- Handle multi-dimensional images (TIFF stacks and CZI files).
- Zoom and pan for detailed annotations.
- Support for multiple classes with customizable colors.
- User-friendly interface with intuitive controls.
- Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires.
- Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?
- Load custom SAM2 pre-trained models for flexible and improved semi-automated annotations.

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

### Loading SAM2 Pre-trained Models

To use SAM2 for semi-automated annotations, you need to download the model and config files:

1. Download the desired SAM2 model file (.pt) and its corresponding config file (.yaml) from the [Segment Anything 2 GitHub repository](https://github.com/facebookresearch/segment-anything-2).
2. For the model file, navigate to the main repository page and download the desired .pt file.
3. For the config file, go to the [sam2_configs folder](https://github.com/facebookresearch/segment-anything-2/tree/main/sam2_configs) and download the corresponding .yaml file.
4. Ensure you download matching pairs, for example:
   - For sam2_hiera_small.pt model, download sam2_hiera_s.yaml config file
   - For sam2_hiera_tiny.pt model, download sam2_hiera_t.yaml config file

Once you have downloaded these files, you can load them in the DigitalSreeni Image Annotator application to use SAM2 for semi-automated annotations.

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
   - To use SAM2 Magic Wand:
     - Click "Load SAM2 Model" and select the config (.yaml) and model (.pt) files you downloaded earlier.
     - Select the SAM2 Magic Wand button and draw a rectangle around your object of interest for automated annotation.
   - Edit existing annotations by double-clicking on them.
   - Save your project using "Save Project" or Ctrl+S.
   - Use "Open Project" or Ctrl+O to load a previously saved project.
   - Click "Import Annotations with Images" to load existing COCO JSON annotations along with their images.
   - Use "Export Annotations" to save annotations in various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC).
   - Access the help documentation by clicking the "Help" button or pressing F1.
   - Explore the interface – you might stumble upon some hidden gems and secret features!

3. Keyboard shortcuts:
   - Ctrl + N: Create a new project
   - Ctrl + O: Open an existing project
   - Ctrl + S: Save the current project
   - Ctrl + W: Close the current project
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
