from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
from PyQt5.QtCore import Qt
from .soft_dark_stylesheet import soft_dark_stylesheet
from .default_stylesheet import default_stylesheet

class HelpWindow(QDialog):
    def __init__(self, dark_mode=False, font_size=10, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
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
            <li>Load custom SAM2 pre-trained models for flexible and improved semi-automated annotations</li>
        </ul>

        <h2>Getting Started</h2>
        <h3>Starting a New Project</h3>
        <ol>
            <li>Click "New Project" or use Ctrl+N to start a new project.</li>
            <li>Click "Add New Images" to import multiple images you want to annotate, including TIFF stacks and CZI files.</li>
            <li>For multi-dimensional images, you'll be prompted to assign dimensions (e.g., T for time, Z for depth, C for channels).</li>
            <li>Use "Add Classes" to define classes of interest.</li>
            <li>Start annotating by selecting a class and using the Polygon, Rectangle Tool, or SAM2 Magic Wand.</li>
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
            <li><strong>Choose a Tool:</strong> Select either the Polygon Tool, Rectangle Tool, or SAM2 Magic Wand.</li>
            <li><strong>Create Annotation:</strong>
                <ul>
                    <li>For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.</li>
                    <li>For Rectangle Tool: Click and drag to create a bounding box.</li>
                    <li>For SAM2 Magic Wand: Ensure a SAM2 model is loaded. Click the SAM2 Magic Wand button to activate assisted annotation. Click and drag around an object, and SAM2 will display the segmented mask. Press Enter to accept the annotation, continue drawing to refine it, or press Escape to exit SAM-assisted annotation.</li>
                </ul>
            </li>
        </ol>

        <h2>Exporting Annotations</h2>
        <ol>
            <li>Click "Export Annotations" to open the export dialog.</li>
            <li>Select the desired export format from the dropdown menu:</li>
            <ul>
                <li><strong>COCO JSON:</strong> Exports a JSON file in COCO format. Save it in the same directory as the images for easy reimport.</li>
                <li><strong>YOLO v8:</strong> Exports txt files for each image with annotations, along with a yaml file, saved in a 'labels' directory.</li>
                <li><strong>Labeled images:</strong> Saves labeled images for each class in separate directories.</li>
                <li><strong>Semantic labels:</strong> Exports semantic label images where each class is represented by a unique pixel value.</li>
                <li><strong>Pascal VOC BBox:</strong> Exports XML files with bounding box annotations in Pascal VOC format.</li>
                <li><strong>Pascal VOC BBox + Segmentation:</strong> Exports XML files with both bounding box and segmentation annotations in Pascal VOC format.</li>
            </ul>
            <li>Choose the export location and confirm to save the annotations in the selected format.</li>
        </ol>

        <h2>Navigation and Viewing</h2>
        <ul>
            <li><strong>Zoom:</strong> Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.</li>
            <li><strong>Pan:</strong> Hold Ctrl, click the left mouse button, and move the mouse.</li>
            <li><strong>Switch Images:</strong> Click on an image name in the image list on the right.</li>
            <li><strong>Navigate Slices:</strong> For multi-dimensional images, use the slice list on the right to click through slices.</li>
        </ul>

        <h2>Keyboard Shortcuts</h2>
        <ul>
            <li><strong>Ctrl + N:</strong> Create a new project</li>
            <li><strong>Ctrl + O:</strong> Open an existing project</li>
            <li><strong>Ctrl + S:</strong> Save the current project</li>
            <li><strong>Ctrl + W:</strong> Close the current project</li>
            <li><strong>F1:</strong> Open this help window</li>
            <li><strong>Ctrl + Wheel:</strong> Zoom in/out</li>
            <li><strong>Esc:</strong> Cancel current annotation, exit edit mode, or exit SAM-assisted annotation</li>
            <li><strong>Enter:</strong> Finish current annotation, exit edit mode, or accept SAM-generated mask</li>
            <li><strong>Up/Down Arrow Keys:</strong> Navigate through slices in multi-dimensional images</li>
        </ul>

        <h2>Known Issues</h2>
        <p>When opening images before loading saved annotations, the annotations may not display correctly for the first image. To avoid this issue, it is recommended to load saved annotations first, followed by opening the corresponding images.</p>

        <h2>Getting Help</h2>
        <p>If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.</p>
        """
        self.text_browser.setHtml(help_text)