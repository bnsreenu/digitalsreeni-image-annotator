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
    version="0.1.5",  # version number
    author="Dr. Sreenivas Bhattiprolu",
    author_email="digitalsreeni@gmail.com",
    description="A tool for annotating images with polygons and rectangles",
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
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "PyQt5>=5.15.7",
    ],
    entry_points={
        "console_scripts": [
            "digitalsreeni-image-annotator=digitalsreeni_image_annotator.main:main",
        ],
    },
)
