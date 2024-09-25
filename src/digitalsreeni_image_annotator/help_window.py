from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
from PyQt5.QtCore import Qt
from .soft_dark_stylesheet import soft_dark_stylesheet
from .default_stylesheet import default_stylesheet

class HelpWindow(QDialog):
    def __init__(self, dark_mode=False, font_size=10):
        super().__init__()
        self.setWindowTitle("Help")
        self.setModal(False)  # Make it non-modal
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        layout.addWidget(self.text_browser)
        self.setLayout(layout)
        
        if dark_mode:
            self.setStyleSheet(soft_dark_stylesheet)
        else:
            self.setStyleSheet(default_stylesheet)
        
        self.font_size = font_size
        self.apply_font_size()
        self.load_help_content()
        
    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()
    
    def apply_font_size(self):
        self.setStyleSheet(f"QWidget {{ font-size: {self.font_size}pt; }}")
        font = self.text_browser.font()
        font.setPointSize(self.font_size)
        self.text_browser.setFont(font)

    def load_help_content(self):
        help_text = """
        <h1>Image Annotator Help Guide</h1>

        <h2>Overview</h2>
        <p>Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in various formats, including COCO-style JSON, YOLO v8, and Pascal VOC. Annotations can be defined using manual tools like the polygon tool or in a semi-automated way with the assistance of the Segment Anything Model (SAM-2) pre-trained model. The tool supports multi-dimensional images such as TIFF stacks and CZI files and provides dark mode and adjustable application font sizes for enhanced GUI experience.</p>

        <h2>Key Features</h2>
        <ul>
            <li>Semi-automated annotations with SAM-2 assistance (Segment Anything Model)</li>
            <li>Manual annotations with polygons and rectangles</li>
            <li>Save and load projects for continued work</li>
            <li>Import existing COCO JSON annotations with images</li>
            <li>Export annotations to various formats (COCO JSON, YOLO v8, Labeled images, Semantic labels, Pascal VOC)</li>
            <li>Handle multi-dimensional images (TIFF stacks and CZI files)</li>
            <li>Zoom and pan for detailed annotations</li>
            <li>Support for multiple classes with customizable colors</li>
            <li>User-friendly interface with intuitive controls</li>
            <li>Adjustable application font size</li>
            <li>Dark mode for comfortable viewing</li>
            <li>Support for common image formats (PNG, JPG, BMP) and multi-dimensional formats (TIFF, CZI)</li>
            <li>Pick appropriate pre-trained SAM2 model for flexible and improved semi-automated annotations</li>
            <li>Additional tools for dataset management and image processing</li>
        </ul>

        <h2>Getting Started</h2>
        <h3>Starting a New Project</h3>
        <ol>
            <li>Click "New Project" or use Ctrl+N to start a new project.</li>
            <li>Click "Add New Images" to import multiple images you want to annotate, including TIFF stacks and CZI files.</li>
            <li>For multi-dimensional images, you'll be prompted to assign dimensions (e.g., T for time, Z for depth, C for channels).</li>
            <li>Use "Add Classes" to define classes of interest.</li>
            <li>Start annotating by selecting a class and using the Polygon, Rectangle Tool, or SAM-Assisted tool.</li>
        </ol>

        <h3>Opening an Existing Project</h3>
        <ol>
            <li>Click "Open Project" or use Ctrl+O to load a previously saved project.</li>
            <li>If there are any missing images, you'll be prompted to locate them on your drive. Located images will be automatically copied to the project directory.</li>
            <li>If you choose not to locate missing images, the annotations for those images will be removed.</li>
        </ol>

        <h3>Importing Existing Annotations</h3>
        <ol>
            <li>Click "Import Annotations with Images" to load existing COCO JSON annotations along with their corresponding images.</li>
            <li>Select the COCO JSON file. The images should be in the same directory as the JSON file.</li>
            <li>The annotations and images will be loaded into your current project.</li>
        </ol>

        <h2>Annotation Process</h2>
        <ol>
            <li><strong>Select a Class:</strong> Choose the class you want to annotate from the class list.</li>
            <li><strong>Choose a Tool:</strong> Select either the Polygon Tool, Rectangle Tool, or SAM-Assisted tool.</li>
            <li><strong>Create Annotation:</strong>
                <ul>
                    <li>For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.</li>
                    <li>For Rectangle Tool: Click and drag to create a bounding box.</li>
                    <li>For SAM-Assisted tool: 
                        <ol>
                            <li>Select a SAM model from the "Pick a SAM Model" dropdown. It's recommended to use smaller models like SAM2 tiny or SAM2 small for better performance.</li>
                            <li>Note: When you select a model for the first time, the application needs to download it. This process may take a few seconds to a minute, depending on your internet connection speed. Subsequent uses of the same model will be faster as it will already be cached locally, in your working directory.</li>
                            <li>Click the "SAM-Assisted" button to activate the tool.</li>
                            <li>Draw a rectangle around objects of interest to allow SAM2 to automatically detect objects.</li>
                            <li>SAM2 will provide various outputs with different scores, and only the top-scoring region will be displayed.</li>
                            <li>If the desired result isn't achieved on the first try, draw again.</li>
                            <li>For low-quality images where SAM2 may not auto-detect objects, manual tools may be necessary.</li>
                        </ol>
                    </li>
                </ul>
            </li>
        </ol>

        <h2>Exporting Annotations</h2>
        <ol>
            <li>Click "Export Annotations" to open the export dialog.</li>
            <li>Select the desired export format from the dropdown menu.</li>
            <li>Choose the export location and confirm to save the annotations in the selected format.</li>
        </ol>

        <h2>Navigation and Viewing</h2>
        <ul>
            <li><strong>Zoom:</strong> Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.</li>
            <li><strong>Pan:</strong> Hold Ctrl, click the left mouse button, and move the mouse.</li>
            <li><strong>Switch Images:</strong> Click on an image name in the image list on the right.</li>
            <li><strong>Navigate Slices:</strong> For multi-dimensional images, use the slice list on the right to click through slices.</li>
        </ul>

        <h2>Tools Menu</h2>
        <p>The Tools menu provides access to various useful tools for dataset management and image processing. Each tool opens an intuitive GUI to guide you through the process:</p>
        <ul>
            <li><strong>Annotation Statistics:</strong> Provides statistical information about your annotations.</li>
            <li><strong>COCO JSON Combiner:</strong> Allows you to combine multiple COCO JSON annotation files.</li>
            <li><strong>Dataset Splitter:</strong> Helps you split your dataset into train, validation, and test sets.</li>
            <li><strong>Stack to Slices:</strong> Converts multi-dimensional image stacks into individual 2D slices.</li>
            <li><strong>Image Patcher:</strong> Splits large images into smaller patches with or without overlap.</li>
            <li><strong>Image Augmenter:</strong> Applies various transformations to augment your image dataset.</li>
        </ul>

        <h2>Keyboard Shortcuts</h2>
        <ul>
            <li><strong>Ctrl + N:</strong> Create a new project</li>
            <li><strong>Ctrl + O:</strong> Open an existing project</li>
            <li><strong>Ctrl + S:</strong> Save the current project</li>
            <li><strong>Ctrl + W:</strong> Close the current project</li>
            <li><strong>Ctrl + Shift + S:</strong> Open Annotation Statistics</li>
            <li><strong>F1:</strong> Open this help window</li>
            <li><strong>Ctrl + Wheel:</strong> Zoom in/out</li>
            <li><strong>Esc:</strong> Cancel current annotation, exit edit mode, or exit SAM-assisted annotation</li>
            <li><strong>Enter:</strong> Finish current annotation, exit edit mode, or accept SAM-generated mask</li>
            <li><strong>Up/Down Arrow Keys:</strong> Navigate through slices in multi-dimensional images</li>
        </ul>

        <h2>Getting Help</h2>
        <p>If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.</p>
        """
        self.text_browser.setHtml(help_text)
