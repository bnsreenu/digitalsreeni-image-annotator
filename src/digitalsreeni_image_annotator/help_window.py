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
        <p>Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation and object detection. It allows users to create, edit, and save annotations in COCO-style JSON format, including both polygon (segmentation) and bounding box information. Annotations can be defined using manual tools like the polygon tool or in a semi-automated way with the assistance of the Segment Anything Model (SAM-2) pre-trained model. The tool now supports multi-dimensional images such as TIFF stacks and CZI files and also provides dark mode and adjustable application font sizes for enhanced GUI experience.</p>

        <h2>Key Features</h2>
        <ul>
            <li>Semi-automated annotations with SAM-2 assistance (Segment Anything Model) — Because who doesn't love a helpful AI sidekick?</li>
            <li>Manual annotations with polygons and rectangles — For when you want to show SAM-2 who's really in charge</li>
            <li>Save annotations in COCO-compatible JSON format</li>
            <li>Edit existing annotations</li>
            <li>Load and continue previous annotation work</li>
            <li>Handle multi-dimensional images (TIFF stacks and CZI files)</li>
            <li>Zoom and pan for detailed annotations</li>
            <li>Support for multiple classes with customizable colors</li>
            <li>Import and export annotations</li>
            <li>User-friendly interface with intuitive controls</li>
            <li>Change the application font size on the fly — Make your annotations as big or small as your caffeine level requires</li>
            <li>Dark mode for those late-night annotation marathons — Who needs sleep when you have dark mode?</li>
            <li>Support for common image formats (PNG, JPG, BMP) and multi-dimensional formats (TIFF, CZI)</li>
            <li>Load custom SAM2 pre-trained models for flexible and improved semi-automated annotations</li>
        </ul>

        <h2>Getting Started</h2>
        <h3>Starting a New Project</h3>
        <ol>
            <li>Click "Open New Image Set" to import multiple images you want to annotate, including TIFF stacks and CZI files.</li>
            <li>For multi-dimensional images, you'll be prompted to assign dimensions (e.g., T for time, Z for depth, C for channels).</li>
            <li>Use "Add Class" to define classes of interest.</li>
            <li>Start annotating by selecting a class and using the Polygon, Rectangle Tool, or SAM2 Magic Wand.</li>
        </ol>

        <h3>Continuing a Previous Project</h3>
        <ol>
            <li>Click "Import Saved Annotations" to load your previous work.</li>
            <li>Use "Open New Image Set" to load the corresponding images.</li>
            <li>If needed, use "Add More Images" to include additional images for annotation.</li>
        </ol>

        <h2>Loading SAM2 Pre-trained Models</h2>
        <ol>
            <li>Download the desired SAM2 model file (.pt) and its corresponding config file (.yaml) from the <a href="https://github.com/facebookresearch/segment-anything-2">Segment Anything 2 GitHub repository</a>.</li>
            <li>For the model file, navigate to the main repository page and download the desired .pt file.</li>
            <li>For the config file, go to the <a href="https://github.com/facebookresearch/segment-anything-2/tree/main/sam2_configs">sam2_configs folder</a> and download the corresponding .yaml file.</li>
            <li>Ensure you download matching pairs, for example:
                <ul>
                    <li>For sam2_hiera_small.pt model, download sam2_hiera_s.yaml config file</li>
                    <li>For sam2_hiera_tiny.pt model, download sam2_hiera_t.yaml config file</li>
                </ul>
            </li>
            <li>In the Image Annotator, click the "Load SAM2 Model" button in the Automated Tools section.</li>
            <li>First, select the config file (.yaml) when prompted.</li>
            <li>Then, select the corresponding model file (.pt) when prompted.</li>
            <li>Once loaded successfully, the SAM2 Magic Wand button will become active for use.</li>
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

        <h2>Navigation and Viewing</h2>
        <ul>
            <li><strong>Zoom:</strong> Use the slider at the bottom of the image, or hold Ctrl and use the mouse wheel.</li>
            <li><strong>Pan:</strong> Hold Ctrl, click the left mouse button, and move the mouse.</li>
            <li><strong>Switch Images:</strong> Click on an image name in the image list on the right.</li>
            <li><strong>Navigate Slices:</strong> For multi-dimensional images, use the slice list on the right to click through slices.</li>
        </ul>

        <h2>Handling Multi-dimensional Images</h2>
        <ul>
            <li>When opening a TIFF stack or CZI file, you'll be prompted to assign dimensions (e.g., T, Z, S, C, H, W).</li>
            <li>The slice list on the right will show all available slices for the current image.</li>
            <li>Annotations are specific to each slice and will be saved accordingly.</li>
            <li>Slices with annotations will be highlighted in green in the slice list.</li>
            <li>You can reassign dimensions after import by right-clicking on the image name. Note that all annotations for that image will be lost when dimensions are changed after importing and annotating.</li>
        </ul>

        <h2>Editing Annotations</h2>
        <ol>
            <li>Ensure you're not in annotation mode (tool buttons should be grey, not blue).</li>
            <li>Double-click an existing annotation to enter edit mode.</li>
            <li>Modify the annotation:
                <ul>
                    <li>Move points: Click and drag existing points.</li>
                    <li>Add points: Click on the boundary line.</li>
                    <li>Delete points: Shift + click on existing points.</li>
                </ul>
            </li>
            <li>Press Enter to accept the edits.</li>
        </ol>

        <h2>Managing Classes</h2>
        <ul>
            <li><strong>Add Class:</strong> Click the "Add Class" button.</li>
            <li><strong>Change Class Color:</strong> Right-click a class, select "Change Color", and choose from the palette.</li>
            <li><strong>Rename Class:</strong> Right-click a class and select "Rename Class".</li>
            <li><strong>Delete Class:</strong> Right-click a class and select "Delete Class".</li>
        </ul>

        <h2>Managing Annotations</h2>
        <ul>
            <li><strong>View Annotations:</strong> Check the Annotations list at the bottom left of the GUI.</li>
            <li><strong>Highlight Annotations:</strong> Select an annotation from the list to highlight it in red on the image.</li>
            <li><strong>Multi-select Annotations:</strong> Hold Ctrl and click multiple annotations.</li>
            <li><strong>Delete Annotations:</strong> Select annotation(s) and click "Delete Selected Annotations".</li>
        </ul>

        <h2>Saving Your Work</h2>
        <ul>
            <li>Click "Save Annotations" to save your work as a COCO-style JSON file.</li>
            <li>For multi-dimensional images, annotated slices will be saved as separate PNG files along with the JSON file.</li>
            <li>You can close the program after saving and continue your work later by importing the saved annotations.</li>
        </ul>

        <h2>Customization</h2>
        <ul>
            <li><strong>Dark Mode:</strong> Click the "Toggle Dark Mode" button to switch between light and dark themes for comfortable viewing in different lighting conditions.</li>
            <li><strong>Font Size:</strong> Use the Font Size drop-down selector to adjust the application's font size to your comfort level. Make your annotations as big or small as your caffeine level requires!</li>
        </ul>

        <h2>Keyboard Shortcuts</h2>
        <ul>
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
