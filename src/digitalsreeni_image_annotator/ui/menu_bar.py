"""Build the application menu bar.

Moved verbatim from ImageAnnotator.create_menu_bar (Phase 8). Every
action's `triggered` connects to a method on `window` (the
ImageAnnotator instance) — many of those are thin delegates to
controllers, but the menu doesn't need to know that.
"""

from PyQt6.QtGui import QAction, QKeySequence

from . import theme


def build_menu_bar(window):
    menu_bar = window.menuBar()

    # Project Menu
    project_menu = menu_bar.addMenu("&Project")

    new_project_action = QAction("&New Project", window)
    new_project_action.setShortcut(QKeySequence.StandardKey.New)
    new_project_action.triggered.connect(window.new_project)
    project_menu.addAction(new_project_action)

    open_project_action = QAction("&Open Project", window)
    open_project_action.setShortcut(QKeySequence.StandardKey.Open)
    open_project_action.triggered.connect(window.open_project)
    project_menu.addAction(open_project_action)

    save_project_action = QAction("&Save Project", window)
    save_project_action.setShortcut(QKeySequence.StandardKey.Save)
    save_project_action.triggered.connect(window.save_project)
    project_menu.addAction(save_project_action)

    save_project_as_action = QAction("Save Project &As...", window)
    save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    save_project_as_action.triggered.connect(window.save_project_as)
    project_menu.addAction(save_project_as_action)

    close_project_action = QAction("&Close Project", window)
    close_project_action.setShortcut(QKeySequence("Ctrl+W"))
    close_project_action.triggered.connect(window.close_project)
    project_menu.addAction(close_project_action)

    project_details_action = QAction("Project &Details", window)
    project_details_action.setShortcut(QKeySequence("Ctrl+I"))
    project_details_action.triggered.connect(window.show_project_details)
    project_menu.addAction(project_details_action)

    search_projects_action = QAction("&Search Projects", window)
    search_projects_action.setShortcut(QKeySequence("Ctrl+F"))
    search_projects_action.triggered.connect(window.show_project_search)
    project_menu.addAction(search_projects_action)

    # Settings Menu
    settings_menu = menu_bar.addMenu("&Settings")

    font_size_menu = settings_menu.addMenu("&Font Size")
    window._font_preset_actions = {}
    for size in ["Small", "Medium", "Large", "XL", "XXL"]:
        action = QAction(size, window)
        action.setCheckable(True)
        action.triggered.connect(lambda checked, s=size: window.change_font_size(s))
        font_size_menu.addAction(action)
        window._font_preset_actions[size] = action
    # Show the persisted size as checked from the first frame (no
    # preset is checked when ui_font_pt sits between preset values).
    theme.sync_font_menu(window)

    font_size_menu.addSeparator()

    # Continuous UI zoom for low-vision users — steps ui_font_pt ±1pt
    # within 8-24. Secondary Ctrl++ / Ctrl+- sequences cover keypads
    # and layouts where Ctrl+Shift+= is awkward.
    increase_font_action = QAction("&Increase Font Size", window)
    increase_font_action.setShortcuts(
        [QKeySequence("Ctrl+Shift+="), QKeySequence("Ctrl++")]
    )
    increase_font_action.triggered.connect(lambda: window.step_font_size(1))
    font_size_menu.addAction(increase_font_action)

    decrease_font_action = QAction("&Decrease Font Size", window)
    decrease_font_action.setShortcuts(
        [QKeySequence("Ctrl+Shift+-"), QKeySequence("Ctrl+-")]
    )
    decrease_font_action.triggered.connect(lambda: window.step_font_size(-1))
    font_size_menu.addAction(decrease_font_action)

    reset_font_action = QAction("&Reset Font Size", window)
    reset_font_action.setShortcut(QKeySequence("Ctrl+Shift+0"))
    reset_font_action.triggered.connect(window.reset_font_size)
    font_size_menu.addAction(reset_font_action)

    toggle_dark_mode_action = QAction("Toggle &Dark Mode", window)
    toggle_dark_mode_action.setShortcut(QKeySequence("Ctrl+D"))
    toggle_dark_mode_action.triggered.connect(window.toggle_dark_mode)
    settings_menu.addAction(toggle_dark_mode_action)

    # Experiment tracking (issue #74) — configure MLflow + open its UI.
    tracking_menu = settings_menu.addMenu("&Experiment Tracking")

    mlflow_settings_action = QAction("MLflow Settings…", window)
    mlflow_settings_action.triggered.connect(window.show_mlflow_settings)
    tracking_menu.addAction(mlflow_settings_action)

    open_mlflow_ui_action = QAction("Open MLflow UI", window)
    open_mlflow_ui_action.triggered.connect(window.open_mlflow_ui)
    tracking_menu.addAction(open_mlflow_ui_action)

    # Tools Menu
    tools_menu = menu_bar.addMenu("&Tools")

    annotation_stats_action = QAction("Annotation Statistics", window)
    annotation_stats_action.triggered.connect(window.show_annotation_statistics)
    annotation_stats_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
    tools_menu.addAction(annotation_stats_action)

    coco_json_combiner_action = QAction("COCO JSON Combiner", window)
    coco_json_combiner_action.triggered.connect(window.show_coco_json_combiner)
    tools_menu.addAction(coco_json_combiner_action)

    dataset_splitter_action = QAction("Dataset Splitter", window)
    dataset_splitter_action.triggered.connect(window.open_dataset_splitter)
    tools_menu.addAction(dataset_splitter_action)

    dino_merge_action = QAction("Merge COCO for Training", window)
    dino_merge_action.triggered.connect(window.show_dino_merge_dialog)
    tools_menu.addAction(dino_merge_action)

    stack_to_slices_action = QAction("Stack to Slices", window)
    stack_to_slices_action.triggered.connect(window.show_stack_to_slices)
    tools_menu.addAction(stack_to_slices_action)

    image_patcher_action = QAction("Image Patcher", window)
    image_patcher_action.triggered.connect(window.show_image_patcher)
    tools_menu.addAction(image_patcher_action)

    image_augmenter_action = QAction("Image Augmenter", window)
    image_augmenter_action.triggered.connect(window.show_image_augmenter)
    tools_menu.addAction(image_augmenter_action)

    slice_registration_action = QAction("Slice Registration", window)
    slice_registration_action.triggered.connect(window.show_slice_registration)
    tools_menu.addAction(slice_registration_action)

    stack_interpolator_action = QAction("Stack Interpolator", window)
    stack_interpolator_action.triggered.connect(window.show_stack_interpolator)
    tools_menu.addAction(stack_interpolator_action)

    dicom_converter_action = QAction("DICOM Converter", window)
    dicom_converter_action.triggered.connect(window.show_dicom_converter)
    tools_menu.addAction(dicom_converter_action)

    export_frames_action = QAction("Export Annotated Video Frames…", window)
    export_frames_action.triggered.connect(window.export_annotated_frames)
    tools_menu.addAction(export_frames_action)

    tools_menu.addSeparator()

    # SAM 3 video object tracking (issue #51). "Track Selected Object" is only
    # meaningful with a video + SAM 3 loaded + one segmentation selected, so its
    # enabled state is refreshed from can_track() each time the menu opens.
    track_object_action = QAction("Track Selected Object…", window)
    track_object_action.triggered.connect(window.track_selected_object)
    tools_menu.addAction(track_object_action)

    undo_track_action = QAction("Undo Last Track", window)
    undo_track_action.triggered.connect(window.undo_last_track)
    tools_menu.addAction(undo_track_action)

    tools_menu.aboutToShow.connect(
        lambda: track_object_action.setEnabled(window.can_track())
    )

    tools_menu.addSeparator()

    unload_models_action = QAction("Unload AI Models (Free GPU Memory)", window)
    unload_models_action.triggered.connect(window.unload_ai_models)
    tools_menu.addAction(unload_models_action)

    # Help Menu
    help_menu = menu_bar.addMenu("&Help")

    help_action = QAction("&Show Help", window)
    help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
    help_action.triggered.connect(window.show_help)
    help_menu.addAction(help_action)
