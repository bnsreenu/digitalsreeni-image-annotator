"""Build the left sidebar, central image area, and right image list.

Moved verbatim from ImageAnnotator (Phase 8). Each builder takes
`window` (the ImageAnnotator instance), attaches widgets as
`window.X = ...` for the references read by other modules, and
connects signals to `window.<method>` (the delegate methods on
ImageAnnotator which forward to controllers).
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.constants import (
    ANNOT_COL_AREA,
    ANNOT_COL_CLASS,
    ANNOT_COL_DETAIL,
    ANNOT_COL_ID,
)
from ..dialogs.dino_phrase_editor import ClassThresholdTable, PhraseEditorPanel


def _section_header(text):
    label = QLabel(text)
    label.setProperty("class", "section-header")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    return label


def build_sidebar(window):
    window.sidebar = QWidget()
    window.sidebar_layout = QVBoxLayout(window.sidebar)
    window.layout.addWidget(window.sidebar, 1)

    # Import functionality
    window.import_button = QPushButton("Import Annotations with Images")
    window.import_button.clicked.connect(window.import_annotations)
    window.sidebar_layout.addWidget(window.import_button)

    window.import_format_selector = QComboBox()
    window.import_format_selector.addItem("COCO JSON")
    window.import_format_selector.addItem("YOLO (v4 and earlier)")
    window.import_format_selector.addItem("YOLO (v5+)")
    window.sidebar_layout.addWidget(window.import_format_selector)

    # Add spacing
    window.sidebar_layout.addSpacing(20)

    window.add_images_button = QPushButton("Add New Images")
    window.add_images_button.clicked.connect(window.add_images)
    window.sidebar_layout.addWidget(window.add_images_button)

    window.add_class_button = QPushButton("Add Classes")
    window.add_class_button.clicked.connect(lambda: window.add_class())
    window.sidebar_layout.addWidget(window.add_class_button)

    # Class list (without the "Classes" header)
    window.class_list = QListWidget()
    window.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    window.class_list.customContextMenuRequested.connect(window.show_class_context_menu)
    window.class_list.itemClicked.connect(window.on_class_selected)
    # itemChanged fires when a class's checkbox is toggled; routes to
    # visibility toggling on the class controller. Previously wired in
    # ImageAnnotator.__init__ post-setup_ui; moved here to live next
    # to the widget construction.
    window.class_list.itemChanged.connect(window.toggle_class_visibility)
    window.sidebar_layout.addWidget(window.class_list)

    # Annotation section
    window.sidebar_layout.addWidget(_section_header("Annotation"))
    annotation_widget = QWidget()
    annotation_layout = QVBoxLayout(annotation_widget)

    # Manual tools subsection
    manual_widget = QWidget()
    manual_layout = QVBoxLayout(manual_widget)

    button_layout_top = QHBoxLayout()
    window.polygon_button = QPushButton("Polygon")
    window.polygon_button.setCheckable(True)
    window.rectangle_button = QPushButton("Rectangle")
    window.rectangle_button.setCheckable(True)
    button_layout_top.addWidget(window.polygon_button)
    button_layout_top.addWidget(window.rectangle_button)

    button_layout_bottom = QHBoxLayout()
    window.paint_brush_button = QPushButton("Paint Brush")
    window.paint_brush_button.setCheckable(True)
    window.eraser_button = QPushButton("Eraser")
    window.eraser_button.setCheckable(True)
    button_layout_bottom.addWidget(window.paint_brush_button)
    button_layout_bottom.addWidget(window.eraser_button)

    button_layout_keypoint = QHBoxLayout()
    window.keypoint_button = QPushButton("Keypoint")
    window.keypoint_button.setCheckable(True)
    window.keypoint_button.setToolTip(
        "Place pose keypoints (issue #35). Define a keypoint schema on the class "
        "first (right-click the class → Define Keypoint Schema)."
    )
    button_layout_keypoint.addWidget(window.keypoint_button)

    manual_layout.addLayout(button_layout_top)
    manual_layout.addLayout(button_layout_bottom)
    manual_layout.addLayout(button_layout_keypoint)

    annotation_layout.addWidget(manual_widget)

    # SAM-Assisted tools subsection
    sam_widget = QWidget()
    sam_layout = QVBoxLayout(sam_widget)

    sam_buttons_layout = QHBoxLayout()

    window.sam_box_button = QPushButton("SAM-box")
    window.sam_box_button.setCheckable(True)
    window.sam_box_button.clicked.connect(window.toggle_sam_box)

    window.sam_points_button = QPushButton("SAM-points")
    window.sam_points_button.setCheckable(True)
    window.sam_points_button.clicked.connect(window.toggle_sam_points)

    sam_buttons_layout.addWidget(window.sam_box_button)
    sam_buttons_layout.addWidget(window.sam_points_button)
    sam_layout.addLayout(sam_buttons_layout)

    # SAM model selector
    window.sam_model_selector = QComboBox()
    window.sam_model_selector.addItem("Pick a SAM Model")
    window.sam_model_selector.addItems(list(window.sam_utils.sam_models.keys()))
    window.sam_model_selector.currentTextChanged.connect(window.change_sam_model)
    sam_layout.addWidget(window.sam_model_selector)

    annotation_layout.addWidget(sam_widget)

    # --- LLM-Assisted Detection (DINO) subsection ---
    dino_widget = QWidget()
    dino_layout = QVBoxLayout(dino_widget)

    window.dino_model_selector = QComboBox()
    window.dino_model_selector.addItem("Pick a DINO Model")
    window.dino_model_selector.addItem("grounding-dino-base")
    window.dino_model_selector.addItem("grounding-dino-tiny")
    window.dino_model_selector.addItem("Custom / fine-tuned (browse)")
    window.dino_model_selector.currentTextChanged.connect(window._on_dino_model_changed)
    dino_layout.addWidget(window.dino_model_selector)

    # Custom model browse row (hidden by default)
    window.dino_browse_row = QWidget()
    dino_browse_layout = QHBoxLayout(window.dino_browse_row)
    dino_browse_layout.setContentsMargins(0, 0, 0, 0)
    window.lbl_dino_custom = QLabel("No path set")
    window.lbl_dino_custom.setWordWrap(True)
    # No inline colour: `palette(text)` resolves to a near-white role in light
    # mode (rendering this caption unreadable). The global QLabel stylesheet
    # rule gives a theme-correct colour in both modes — leave the property out
    # so the global sheet wins. See "No Hardcoded Colors Rule" in CLAUDE.md.
    # No font-size here either — the caption follows the global ui_font_pt rule.
    btn_dino_browse = QPushButton("Browse")
    # No fixed width — a 60px cap clipped the caption at large UI font
    # sizes (low-vision zoom); sizeHint tracks the active font.
    btn_dino_browse.clicked.connect(window.browse_dino_model)
    dino_browse_layout.addWidget(window.lbl_dino_custom, 1)
    dino_browse_layout.addWidget(btn_dino_browse)
    window.dino_browse_row.setVisible(False)
    dino_layout.addWidget(window.dino_browse_row)

    window.lbl_dino_status = QLabel("No DINO model loaded")
    window.lbl_dino_status.setWordWrap(True)
    # No hardcoded background — let the active stylesheet (light or
    # dark) provide it via QLabel rules. Hardcoded #f5f5f5 used to
    # punch a bright rectangle into the dark sidebar.
    window.lbl_dino_status.setStyleSheet(
        "padding:4px;border-radius:3px;"
        "border:1px solid palette(mid);"
    )
    dino_layout.addWidget(window.lbl_dino_status)

    # Threshold table
    window.dino_class_table = ClassThresholdTable()
    window.dino_class_table.itemSelectionChanged.connect(
        window.on_dino_class_row_changed
    )
    dino_layout.addWidget(window.dino_class_table)

    # Phrase editor
    window.dino_phrase_panel = PhraseEditorPanel()
    dino_layout.addWidget(window.dino_phrase_panel)

    # Detect buttons
    det_btn_layout = QHBoxLayout()
    window.btn_detect_single = QPushButton("Detect Current Image")
    window.btn_detect_single.clicked.connect(window.run_dino_detection_single)
    window.btn_detect_single.setEnabled(False)
    det_btn_layout.addWidget(window.btn_detect_single)

    window.btn_detect_batch = QPushButton("Detect All Images")
    window.btn_detect_batch.clicked.connect(window.run_dino_detection_batch)
    window.btn_detect_batch.setEnabled(False)
    det_btn_layout.addWidget(window.btn_detect_batch)
    dino_layout.addLayout(det_btn_layout)

    # Batch mode
    window.dino_batch_mode = QComboBox()
    window.dino_batch_mode.addItem("Review before accepting")
    window.dino_batch_mode.addItem("Auto-accept all detections")
    dino_layout.addWidget(window.dino_batch_mode)

    annotation_layout.addWidget(dino_widget)
    # --- END DINO section ---

    # Tool group — must include all checkable tool buttons so
    # update_ui_for_current_tool / enable_tools / disable_tools can
    # iterate.
    window.tool_group = QButtonGroup(window)
    window.tool_group.setExclusive(False)
    window.tool_group.addButton(window.polygon_button)
    window.tool_group.addButton(window.rectangle_button)
    window.tool_group.addButton(window.paint_brush_button)
    window.tool_group.addButton(window.eraser_button)
    window.tool_group.addButton(window.keypoint_button)
    window.tool_group.addButton(window.sam_box_button)
    window.tool_group.addButton(window.sam_points_button)

    window.polygon_button.clicked.connect(window.toggle_tool)
    window.rectangle_button.clicked.connect(window.toggle_tool)
    window.paint_brush_button.clicked.connect(window.toggle_tool)
    window.eraser_button.clicked.connect(window.toggle_tool)
    window.keypoint_button.clicked.connect(window.toggle_tool)

    # Annotations table subsection (issue #24). A 4-column table — ID | Class |
    # Area | Detail % — mirroring the DINO ClassThresholdTable idiom (per-row
    # spinbox via setCellWidget, stylesheet-only header). Column 0 holds the
    # annotation dict in UserRole for the value-equality selection bridge.
    annotation_layout.addWidget(QLabel("Annotations"))
    table = QTableWidget(0, 4)
    table.setHorizontalHeaderLabels(["ID", "Class", "Area", "Detail %"])
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    header = table.horizontalHeader()
    header.setSectionResizeMode(ANNOT_COL_CLASS, QHeaderView.ResizeMode.Stretch)
    for col in (ANNOT_COL_ID, ANNOT_COL_AREA, ANNOT_COL_DETAIL):
        header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
    # Structural only — header colours come from the active stylesheet's
    # QHeaderView::section rule (No Hardcoded Colors Rule), so it matches both
    # themes. See dino_phrase_editor.ClassThresholdTable.
    table.setStyleSheet("QHeaderView::section { font-weight: bold; padding: 2px; }")
    window.annotation_list = table
    window.annotation_list.itemSelectionChanged.connect(
        window.update_highlighted_annotations
    )
    annotation_layout.addWidget(window.annotation_list)

    # Sort buttons
    sort_button_layout = QHBoxLayout()
    window.sort_by_class_button = QPushButton("Sort by Class")
    window.sort_by_class_button.clicked.connect(window.sort_annotations_by_class)
    sort_button_layout.addWidget(window.sort_by_class_button)

    window.sort_by_area_button = QPushButton("Sort by Area")
    window.sort_by_area_button.clicked.connect(window.sort_annotations_by_area)
    sort_button_layout.addWidget(window.sort_by_area_button)

    annotation_layout.addLayout(sort_button_layout)

    # Delete / Merge / Change Class buttons
    window.delete_button = QPushButton("Delete")
    window.delete_button.clicked.connect(window.delete_selected_annotations)
    window.merge_button = QPushButton("Merge")
    window.merge_button.clicked.connect(window.merge_annotations)
    window.change_class_button = QPushButton("Change Class")
    window.change_class_button.clicked.connect(window.change_annotation_class)

    button_layout = QHBoxLayout()
    button_layout.addWidget(window.delete_button)
    button_layout.addWidget(window.merge_button)
    button_layout.addWidget(window.change_class_button)
    annotation_layout.addLayout(button_layout)

    # Export format selector
    window.export_format_selector = QComboBox()
    window.export_format_selector.addItem("COCO JSON")
    window.export_format_selector.addItem("YOLO (v4 and earlier)")
    window.export_format_selector.addItem("YOLO (v5+)")
    window.export_format_selector.addItem("Labeled Images")
    window.export_format_selector.addItem("Semantic Labels")
    window.export_format_selector.addItem("Pascal VOC (BBox)")
    window.export_format_selector.addItem("Pascal VOC (BBox + Segmentation)")

    annotation_layout.addWidget(QLabel("Export Format:"))
    annotation_layout.addWidget(window.export_format_selector)

    window.export_button = QPushButton("Export Annotations")
    window.export_button.clicked.connect(window.export_annotations)
    annotation_layout.addWidget(window.export_button)

    window.sidebar_layout.addWidget(annotation_widget)


def build_image_area(window):
    window.image_widget = QWidget()
    window.image_layout = QVBoxLayout(window.image_widget)
    window.layout.addWidget(window.image_widget, 3)

    window.scroll_area = QScrollArea()
    window.scroll_area.setWidgetResizable(True)
    window.scroll_area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    window.scroll_area.setVerticalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )

    # Use the already initialized image_label
    window.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.scroll_area.setWidget(window.image_label)

    window.image_layout.addWidget(window.scroll_area)

    window.zoom_slider = QSlider(Qt.Orientation.Horizontal)
    window.zoom_slider.setMinimum(10)
    window.zoom_slider.setMaximum(500)
    window.zoom_slider.setValue(100)
    window.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    window.zoom_slider.setTickInterval(50)
    window.zoom_slider.valueChanged.connect(window.zoom_image)
    window.image_layout.addWidget(window.zoom_slider)

    window.image_info_label = QLabel()
    window.image_layout.addWidget(window.image_info_label)


def build_image_list(window):
    window.image_list_widget = QWidget()
    window.image_list_layout = QVBoxLayout(window.image_list_widget)
    window.layout.addWidget(window.image_list_widget, 1)

    window.image_list_label = QLabel("Images:")
    window.image_list_layout.addWidget(window.image_list_label)

    # Annotation-status filter (upstream issue #27). Index order matters:
    # ImageController.apply_image_filter maps 0=all, 1=without, 2=with.
    window.image_filter_combo = QComboBox()
    window.image_filter_combo.addItem("All images")
    window.image_filter_combo.addItem("Without annotations")
    window.image_filter_combo.addItem("With annotations")
    window.image_filter_combo.setToolTip(
        "Filter the image list by annotation status"
    )
    window.image_filter_combo.currentIndexChanged.connect(
        lambda _index: window.apply_image_filter()
    )
    window.image_list_layout.addWidget(window.image_filter_combo)

    window.image_list = QListWidget()
    window.image_list.itemClicked.connect(window.switch_image)
    window.image_list.currentRowChanged.connect(
        lambda row: window.switch_image(window.image_list.currentItem())
    )
    window.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    window.image_list.customContextMenuRequested.connect(window.show_image_context_menu)
    window.image_list_layout.addWidget(window.image_list)

    window.clear_all_button = QPushButton("Clear All Images and Annotations")
    window.clear_all_button.clicked.connect(window.clear_all)
    window.image_list_layout.addWidget(window.clear_all_button)
