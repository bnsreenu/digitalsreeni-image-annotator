"""
Setup file for the DigitalSreeni Image Annotator package.
@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="digitalsreeni-image-annotator",
    version="0.9.0",  # PyQt6 + in-process inference
    author="Dr. Sreenivas Bhattiprolu",
    author_email="digitalsreeni@gmail.com",
    description="A tool for annotating images using manual and automated tools, supporting multi-dimensional images and SAM2-assisted annotations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bnsreenu/digitalsreeni-image-annotator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.10",
    install_requires=[
        "PyQt6>=6.7.0",
        "numpy>=2.0.0",  # pip resolves 2.4+ on Py3.14, 2.2.x on Py3.10 (last 3.10-compatible)
        "Pillow>=10.0.0",
        "tifffile>=2023.0.0",
        "imagecodecs>=2023.1.23",  # tifffile needs it for LZW/compressed TIFF (#56)
        "czifile>=2019.7.2",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0.0",
        "scikit-image>=0.21.0",
        "ultralytics>=8.0.0",
        "plotly>=5.0.0",
        "shapely>=2.0.0",
        "pystackreg>=0.2.7",
        "pydicom>=2.3.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.20.0",
        # MLflow experiment tracking for model training (issue #74). A core
        # dependency, not optional: every SAM/YOLO training run is tracked.
        "mlflow>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "digitalsreeni-image-annotator=digitalsreeni_image_annotator.main:main",
            "sreeni=digitalsreeni_image_annotator.main:main",
        ],
    },
)