from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
from ..ui.soft_dark_stylesheet import soft_dark_stylesheet
from ..ui.default_stylesheet import default_stylesheet

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
        
        self._base_stylesheet = soft_dark_stylesheet if dark_mode else default_stylesheet

        self.font_size = font_size
        self.apply_font_size()
        self.load_help_content()
        
    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()
    
    def apply_font_size(self):
        # Append the font rule to the theme stylesheet — replacing the
        # whole sheet here used to wipe the dark/light theme.
        self.setStyleSheet(
            f"{self._base_stylesheet}\nQWidget {{ font-size: {self.font_size}pt; }}"
        )
        font = self.text_browser.font()
        font.setPointSize(self.font_size)
        self.text_browser.setFont(font)

    def load_help_content(self):
        help_text = """
        <h1>Image Annotator Help Guide</h1>

        <h2>Overview</h2>
        <p>Image Annotator is a user-friendly GUI tool designed for generating masks for image segmentation, object detection, and keypoint/pose annotation. It allows users to create, edit, and save annotations in various formats, including COCO-style JSON, YOLO v8/v11 (including YOLO-pose), and Pascal VOC. Annotations can be defined using manual tools like the polygon tool, or in a semi-automated way with the assistance of the Segment Anything Model (SAM-2 / SAM 3) or Grounding-DINO text-prompted detection. The tool supports multi-dimensional images (TIFF stacks, CZI files) and videos (MP4/AVI/MOV, with SAM 3 object tracking across frames), and provides dark mode, undo/redo, and adjustable application font sizes for enhanced GUI experience.</p>

        <h2>Key Features</h2>
        <ul>
            <li>Semi-automated annotations with SAM-2 assistance (points / box prompts)</li>
            <li>Grounding-DINO and SAM 3 text-prompted detection — describe an object in plain English and review/accept the results for one image or a whole batch</li>
            <li>Fine-tune SAM 2 on your own annotations (with automatic MLflow experiment tracking) instead of relying on the generic pre-trained weights</li>
            <li>Video support (MP4/AVI/MOV): frame-by-frame annotation with a scrub timeline, plus SAM 3 object tracking to propagate a mask across frames</li>
            <li>Keypoint / pose annotation with a per-class named skeleton (COCO instance model), including COCO-keypoints and YOLO-pose export/import and training</li>
            <li>Manual annotations with polygons, rectangles, paint brush, and eraser tools</li>
            <li>Undo / redo for every annotation edit (Ctrl+Z / Ctrl+Y)</li>
            <li>Canvas selection unified with the annotations table, with handle-based resize/move and vertex editing for any selected shape</li>
            <li>Save and load projects for continued work, with autosave</li>
            <li>Import existing COCO JSON / YOLO annotations with images</li>
            <li>Export annotations to various formats (COCO JSON, YOLO v8/v11, YOLO-pose, Labeled images, Semantic labels, Pascal VOC)</li>
            <li>Handle multi-dimensional images (TIFF stacks and CZI files)</li>
            <li>Zoom and pan for detailed annotations</li>
            <li>Support for multiple classes with customizable colors, and per-class visibility toggles</li>
            <li>YOLO training (detection, segmentation, and pose) on your current annotations, with automatic MLflow tracking, and prediction with the trained model</li>
            <li>Adjustable application font size and dark mode</li>
            <li>Additional tools for dataset management and image processing (see the Tools Menu section below)</li>
        </ul>

        <h2>Getting Started</h2>
        <h3>Starting a New Project</h3>
        <ol>
            <li>Click "New Project" or use Ctrl+N to start a new project.</li>
            <li>Click "Add Images / Videos" to import images (including TIFF stacks and CZI files) or videos (MP4/AVI/MOV).</li>
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
            <li><strong>Choose a Tool:</strong> Select the Polygon Tool, Rectangle Tool, or one of the SAM-assisted tools (SAM-box / SAM-points).</li>
            <li><strong>Create Annotation:</strong>
                <ul>
                    <li>For Polygon Tool: Click around the object to define its boundary. Press Enter or click "Finish Polygon" when done.</li>
                    <li>For Rectangle Tool: Click and drag to create a bounding box.</li>
                    <li>For SAM-assisted tools (SAM-box / SAM-points):
                        <ol>
                            <li>Select a SAM model from the "Pick a SAM Model" dropdown. It's recommended to use smaller models like SAM2 tiny or SAM2 small for better performance.</li>
                            <li>Note: When you select a model for the first time, the application needs to download it. This process may take a few seconds to a minute, depending on your internet connection speed. Subsequent uses of the same model will be faster as it will already be cached locally, in your working directory.</li>
                            <li>Click the "SAM-box" button and draw a rectangle around an object of interest, or click the "SAM-points" button and left-click points inside the object (right-click adds negative points to exclude regions).</li>
                            <li>SAM2 will display the top-scoring mask as a temporary prediction. Press Enter to accept it or Esc to discard it.</li>
                            <li>If the desired result isn't achieved on the first try, draw the box again or adjust the points.</li>
                            <li>For low-quality images where SAM2 may not auto-detect objects, manual tools may be necessary.</li>
                        </ol>
                    </li>
                </ul>
            </li>
        </ol>

        <h2>Text-Prompted Detection (Grounding-DINO / SAM 3)</h2>
        <p>Instead of clicking points or drawing a box, you can describe what you want in plain English. In the DINO panel, pick a model from the dropdown: <strong>grounding-dino-base</strong>, <strong>grounding-dino-tiny</strong>, <strong>SAM 3 (text prompt)</strong>, or a custom/fine-tuned checkpoint.</p>
        <ul>
            <li>Add one or more <strong>phrases</strong> per class (e.g. class "drone" &rarr; phrases "drone", "quadcopter") and tune the per-class Box/Text/NMS thresholds.</li>
            <li><strong>Detect Current Image</strong> or <strong>Detect All Images</strong> runs detection (Grounding-DINO boxes are automatically refined into masks by SAM 2; SAM 3 produces masks directly).</li>
            <li>Results show as an orange preview. Press <strong>Enter</strong> to accept them into your annotations, or <strong>Esc</strong> to discard them &mdash; this works even if focus is on a list or button, not just the canvas.</li>
            <li>The "Review before accepting" / "Auto-accept all detections" dropdown controls both the single-image and batch buttons.</li>
            <li><strong>SAM 3 setup note:</strong> its weights file (<code>sam3.pt</code>, ~3.45&nbsp;GB) is gated on Hugging Face and is never auto-downloaded. Request access, then place <code>sam3.pt</code> in the app's working directory or its <code>models/sam/</code> folder before selecting it in the dropdown.</li>
        </ul>

        <h2>Video Annotation and Object Tracking</h2>
        <p>Load an <strong>.mp4 / .avi / .mov</strong> file the same way as an image &mdash; it's treated as a stack whose frames are its slices, decoded on demand. A scrub timeline appears below the canvas; drag it, or use Up/Down (Home/End for first/last frame) on the slice list, to move between frames. There is no autoplay &mdash; annotation is frame-by-frame.</p>
        <ul>
            <li>Every manual and AI-assisted tool works on a video frame exactly as it does on a still image.</li>
            <li>With the <strong>SAM 3</strong> model loaded and exactly one mask annotation selected on the current frame, use <strong>Tools &rarr; Track Selected Object&hellip;</strong> to propagate that mask across the rest of the video. Confident frames are committed automatically; uncertain ones are queued for the same Enter/Escape review used for DINO detections.</li>
            <li><strong>Tools &rarr; Undo Last Track</strong> removes an entire tracking run in one action; individual tracked frames are also undoable with Ctrl+Z.</li>
            <li><strong>Tools &rarr; Export Annotated Video Frames&hellip;</strong> saves every annotated frame of the current video as a PNG.</li>
        </ul>

        <h2>Keypoint / Pose Annotation</h2>
        <p>Mark a named skeleton on instances of a class (people, animals, tools&hellip;), following the COCO pose model.</p>
        <ol>
            <li>Right-click a class &rarr; <strong>Define Keypoint Schema</strong> &rarr; add each point's name, its optional horizontal-flip partner, and the skeleton edges between points.</li>
            <li>Activate the <strong>Keypoint</strong> tool and place the class's points in schema order: left-click for a visible point, right-click (or Shift+left-click) for an occluded one, Backspace to remove the last point, Enter to finish early.</li>
            <li>With an instance selected, drag a point to move it, right-click a point to toggle visible/occluded, or drag the instance's box handles to move/resize the whole pose.</li>
            <li>Pose classes export to and import from COCO-keypoints and YOLO-pose formats, and can be trained/predicted with YOLO exactly like detection or segmentation classes.</li>
        </ol>

        <h2>Selecting, Editing, and Undo/Redo</h2>
        <p>When no drawing tool is active, the canvas is a selection pointer &mdash; and it's unified with the Annotations table below it.</p>
        <ul>
            <li><strong>Click</strong> a mask to select it; <strong>Shift+Click</strong> toggles it in/out of the selection; <strong>drag</strong> on empty canvas rubber-band selects, <strong>Shift+drag</strong> adds to the selection.</li>
            <li>With exactly <strong>one</strong> shape selected, eight square handles appear: drag a handle to resize, drag inside the shape to move it. Double-click a polygon to enter vertex-edit mode.</li>
            <li><strong>Delete</strong> removes the selected annotation(s) instantly &mdash; there's no confirmation dialog, because&hellip;</li>
            <li><strong>Ctrl+Z</strong> undoes the last edit and <strong>Ctrl+Y</strong> (or Ctrl+Shift+Z) redoes it. This covers drawing, deleting, merging, resizing/moving, vertex edits, and every AI-assisted accept.</li>
        </ul>

        <h2>SAM 2 Fine-Tuning and YOLO Training</h2>
        <p>If the built-in SAM 2 weights don't segment your imagery well, fine-tune SAM 2 on your own annotations via <strong>SAM Fine-Tune (beta) &rarr; Train on Current Project&hellip;</strong> &mdash; configure epochs, learning rate, prompt type, and train/validation split, then use the resulting <strong>&#9733; &lt;model name&gt;</strong> entry in the SAM model dropdown like any built-in model.</p>
        <p>Similarly, <strong>YOLO (beta) &rarr; Training &rarr; Train Model</strong> trains a detection, segmentation, or pose model on your current annotations, then <strong>Predict with YOLO Model</strong> runs it on new images for review/accept.</p>
        <p>Every SAM fine-tuning and YOLO training run is automatically tracked in <strong>MLflow</strong> (no toggle needed) &mdash; see <strong>Settings &rarr; Experiment Tracking</strong> to change the tracking store location or open the MLflow dashboard.</p>

        <h2>Exporting Annotations</h2>
        <ol>
            <li>Click "Export Annotations" to open the export dialog.</li>
            <li>Select the desired export format from the dropdown menu (COCO JSON, YOLO v8/v11 including YOLO-pose, Labeled images, Semantic labels, Pascal VOC).</li>
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
            <li><strong>Merge COCO for Training:</strong> Merges accumulated Grounding-DINO/SAM annotations across images into one training-ready COCO JSON.</li>
            <li><strong>Stack to Slices:</strong> Converts multi-dimensional image stacks into individual 2D slices.</li>
            <li><strong>Image Patcher:</strong> Splits large images into smaller patches with or without overlap.</li>
            <li><strong>Image Augmenter:</strong> Applies various transformations to augment your image dataset.</li>
            <li><strong>Slice Registration:</strong> Aligns images in a stack with advanced registration options. </li>
            <li><strong>Stack Interpolator:</strong> Adjusts Z-spacing in image stacks. Excellent tool to generate isotropic volumes from non-isotropic data.</li>
            <li><strong>DICOM Converter:</strong> Converts DICOM files to TIFF format. </li>
            <li><strong>Export Annotated Video Frames&hellip;:</strong> Saves every annotated frame of the current video as a PNG.</li>
            <li><strong>Track Selected Object&hellip; / Undo Last Track:</strong> SAM 3 video object tracking &mdash; see Video Annotation above.</li>
            <li><strong>Unload AI Models (Free GPU Memory):</strong> Drops cached SAM/DINO/SAM 3 models from memory; re-select a model to use AI tools again.</li>
        </ul>

        <h2>Keyboard Shortcuts</h2>
        <ul>
            <li><strong>Ctrl + N:</strong> Create a new project</li>
            <li><strong>Ctrl + O:</strong> Open an existing project</li>
            <li><strong>Ctrl + S:</strong> Save the current project</li>
            <li><strong>Ctrl + Shift + S:</strong> Save Project As&hellip;</li>
            <li><strong>Ctrl + W:</strong> Close the current project</li>
            <li><strong>Ctrl + I:</strong> Project Details</li>
            <li><strong>Ctrl + F:</strong> Search Projects</li>
            <li><strong>Ctrl + Alt + S:</strong> Open Annotation Statistics</li>
            <li><strong>Ctrl + D:</strong> Toggle dark mode</li>
            <li><strong>Ctrl + Z:</strong> Undo the last annotation edit</li>
            <li><strong>Ctrl + Y (or Ctrl + Shift + Z):</strong> Redo</li>
            <li><strong>F1:</strong> Open this help window</li>
            <li><strong>Ctrl + Wheel:</strong> Zoom in/out</li>
            <li><strong>Ctrl + Drag:</strong> Pan the image</li>
            <li><strong>Ctrl + Shift + = (or Ctrl + +):</strong> Increase application font size</li>
            <li><strong>Ctrl + Shift + - (or Ctrl + -):</strong> Decrease application font size</li>
            <li><strong>Ctrl + Shift + 0:</strong> Reset application font size</li>
            <li><strong>Esc:</strong> Cancel current annotation and return to selection mode</li>
            <li><strong>Enter:</strong> Finish/accept current annotation, edit, or AI-generated prediction</li>
            <li><strong>Delete:</strong> Delete the selected annotation(s) on the canvas (undoable)</li>
            <li><strong>Up/Down Arrow Keys:</strong> Navigate through slices or video frames</li>
            <li><strong>Home/End:</strong> Jump to the first/last frame of a video</li>
            <li><strong>- and =:</strong> Adjust paint brush / eraser size</li>
        </ul>

        <h2>Getting Help</h2>
        <p>This window covers the essentials. For a comprehensive, in-depth reference covering every feature in detail, see <strong>USER_MANUAL.md</strong> included with the application (in the project's source/installation folder, and on the project's GitHub repository).</p>
        <p>If you encounter any issues or have suggestions for improvement, please open an issue on our GitHub repository or contact the development team.</p>
        """
        self.text_browser.setHtml(help_text)
