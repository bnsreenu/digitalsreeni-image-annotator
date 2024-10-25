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
    version="0.8.2",  # Updated version number
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
    ],
    python_requires=">=3.10",
    install_requires=[
        "PyQt5>=5.15.7",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "tifffile>=2023.3.15",
        "czifile>=2019.7.2",
        "opencv-python>=4.10.0",
        "pyyaml>=6.0.2",
        "scikit-image>=0.24.0",
        "ultralytics>=8.2.94",
        "plotly>=5.24.1",
        "shapely>=2.0.6",
    ],
    entry_points={
        "console_scripts": [
            "digitalsreeni-image-annotator=digitalsreeni_image_annotator.main:main",
            "sreeni=digitalsreeni_image_annotator.main:main", 
        ],
    },
)