"""
Help Window module for the Image Annotator application.

This module contains the HelpWindow class, which displays the help information
in a new window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextDocument

HELP_TEXT = """
# Image Annotator Help Guide

## Overview

Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in COCO-style JSON format, including both polygon (segmentation) and bounding box information.

## Key Features

- Easy-to-use graphical interface
- Support for polygon and rectangle annotations
- COCO-style JSON output with segmentation and bounding box data
- Ability to load and continue previous annotation work
- Support for multiple image formats (png, jpg, bmp)
- Multi-class annotation support with customizable colors

## Getting Started

### Starting a New Project

1. Click "Open New Image Set" to import multiple images you want to annotate.
2. Use "Add Class" to define classes of interest.
3. Start annotating by selecting a class and using the Polygon or Rectangle Tool.

### Continuing a Previous Project

1. Click "Import Saved Annotations" to load your previous work.
2. Use "Open New Image Set" to load the corresponding images.
3. If needed, use "Add More Images" to include additional images for annotation.

## Annotation Process

1. **Select a Class**: Choose the class you want to annotate from the class list.
2. **Choose a Tool**: Select either the Polygon Tool (recommended for better control) or Rectangle Tool.
3. **Create Annotation**: 
   - For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.
   - For Rectangle Tool: Click and drag to create a bounding box.
4. **Finish Annotation**: Hit Enter or click "Finish Polygon" to complete the annotation.

## Navigation and Viewing

- **Zoom**: Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.
- **Pan**: Hold Ctrl, click the left mouse button, and move the mouse.
- **Switch Images**: Click on an image name in the image list on the right.

## Editing Annotations

1. Ensure you're not in annotation mode (tool buttons should be grey, not blue).
2. Double-click an existing annotation to enter edit mode.
3. Modify the annotation:
   - Move points: Click and drag existing points.
   - Add points: Click on the boundary line.
   - Delete points: Shift + click on existing points.
4. Press Enter to accept the edits.

## Managing Classes

- **Add Class**: Click the "Add Class" button.
- **Change Class Color**: Right-click a class, select "Change Color", and choose from the palette.
- **Rename Class**: Right-click a class and select "Rename Class".
- **Delete Class**: Right-click a class and select "Delete Class".

## Managing Annotations

- **View Annotations**: Check the Annotations list at the bottom left of the GUI.
- **Highlight Annotations**: Select an annotation from the list to highlight it in red on the image.
- **Multi-select Annotations**: Hold Ctrl and click multiple annotations.
- **Delete Annotations**: Select annotation(s) and click "Delete Selected Annotations".

## Saving Your Work

- Click "Save Annotations" to save your work as a COCO-style JSON file.
- You can close the program after saving and continue your work later by importing the saved annotations.

## Supported Formats

- Supports common image formats: PNG, JPG, BMP
- Does not currently support TIFF or other proprietary formats

## Known Issues

- There is a minor bug where annotations may not display for the first image if images are opened before importing annotations. We're working on fixing this issue.

## Getting Help

If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.
"""

class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Annotator Help')
        self.setGeometry(100, 100, 800, 600)
    
        layout = QVBoxLayout()
    
        self.textBrowser = QTextBrowser()
        self.textBrowser.setOpenExternalLinks(True)
        
        # Use setHtml instead of setMarkdown
        doc = QTextDocument()
        doc.setMarkdown(HELP_TEXT)
        self.textBrowser.setDocument(doc)
    
        font = QFont('Arial', 10)
        self.textBrowser.setFont(font)
    
        layout.addWidget(self.textBrowser)
    
        closeButton = QPushButton('Close')
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton, alignment=Qt.AlignRight)
    
        self.setLayout(layout)
