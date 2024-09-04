"""
Setup file for the Image Annotator package.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="image_annotator",
    version="0.1.0",
    author="Dr. Sreenivas Bhattiprolu",
    author_email="your.email@example.com",
    description="A tool for annotating images with polygons and rectangles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image_annotator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "PyQt5>=5.15.0",
    ],
    entry_points={
        "console_scripts": [
            "image_annotator=image_annotator.main:main",
        ],
    },
)