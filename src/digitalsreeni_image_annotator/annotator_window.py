import os
import warnings

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTextEdit,
    QWidget,
)

from .app_settings import load_ui_prefs
from .controllers import io_controller
from .controllers.annotation_controller import AnnotationController
from .controllers.class_controller import ClassController
from .controllers.dino_controller import DINOController
from .controllers.image_controller import ImageController
from .controllers.project_controller import ProjectController
from .controllers.sam_controller import SAMController
from .controllers.sam_train_controller import SAMTrainController
from .controllers.yolo_controller import YOLOController
from .core import image_utils
from .ui import theme
from .ui.menu_bar import build_menu_bar
from .ui.shortcuts import install_event_filters, install_shortcuts
from .ui.sidebar import build_image_area, build_image_list, build_sidebar
from .dialogs.annotation_statistics import show_annotation_statistics
from .dialogs.coco_json_combiner import show_coco_json_combiner
from .dialogs.dino_phrase_editor import ClassThresholdTable, PhraseEditorPanel
from .inference.dino_utils import DINOUtils, GDINO_MODEL_PATHS
from .dialogs.dataset_splitter import DatasetSplitterTool
from .dialogs.dicom_converter import DicomConverter
from .dialogs.dino_merge_dialog import show_dino_merge_dialog
from .dialogs.help_window import HelpWindow
from .dialogs.image_augmenter import show_image_augmenter
from .widgets.canvas_context import CanvasContext
from .widgets.image_label import ImageLabel
from .dialogs.image_patcher import show_image_patcher
from .inference.sam_utils import SAMUtils
from .dialogs.slice_registration import SliceRegistrationTool
from .dialogs.snake_game import SnakeGame
from .dialogs.stack_interpolator import StackInterpolator
from .dialogs.stack_to_slices import show_stack_to_slices

warnings.filterwarnings("ignore", category=UserWarning)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_loading_project = False
        self.backup_project_path = None

        self.project_controller = ProjectController(self)
        self.image_controller = ImageController(self)

        self.setWindowTitle("Image Annotator")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize image_label early — setup_ui's sidebar/image-area
        # builders expect it to exist.
        self.image_label = ImageLabel()

        self.image_label.sam_box_active = False
        self.image_label.sam_points_active = False
        self.image_label.sam_positive_points = []
        self.image_label.sam_negative_points = []

        # Initialize attributes
        self.current_image = None
        self.current_class = None
        self.image_file_name = ""
        self.all_annotations = {}
        self.all_images = []
        self.image_paths = {}
        self.loaded_json = None
        self.class_mapping = {}
        # Per-class keypoint schema for pose classes (issue #35). Keyed by
        # class name; a class is a "pose class" iff it has an entry here.
        # value: {"names": [...], "skeleton": [[a, b], ...], "flip_idx": [...]}
        self.keypoint_schemas = {}
        self.editing_mode = False
        self.current_slice = None
        self.slices = []
        self.current_stack = None
        self.image_dimensions = {}
        self.image_slices = {}
        self.image_shapes = {}

        
        # For paint brush and eraser
        self.paint_brush_size = 10
        self.eraser_size = 10
        # Initialize SAM utils
        self.current_sam_model = None
        self.sam_utils = SAMUtils()

        # Initialize DINO utils for LLM-assisted detection.
        # Phrases and thresholds are owned by the widgets (PhraseEditorPanel
        # and ClassThresholdTable); the project save/load reads/writes them
        # through the widget APIs, not through a shadow dict on self.
        self.dino_utils = DINOUtils()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None

        # Debounce timer for SAM points: wait 1s after last click before inference
        self.sam_inference_timer = QTimer(self)
        self.sam_inference_timer.setSingleShot(True)
        self.sam_inference_timer.timeout.connect(self.apply_sam_prediction)

        # Guards against re-entrant `apply_sam_prediction` calls — the
        # debounce timer can fire while an earlier inference is still
        # pumping inside _run_sync. See apply_sam_prediction().
        self._sam_inference_in_flight = False

        self.sam_controller = SAMController(self)
        self.sam_train_controller = SAMTrainController(self)
        self.dino_controller = DINOController(self)
        self.yolo_controller = YOLOController(self)
        self.annotation_controller = AnnotationController(self)
        self.class_controller = ClassController(self)

        # CanvasContext gives ImageLabel a narrow read view of main-window
        # state. All write paths from the canvas leave as Qt signals
        # connected to controllers below.
        self.image_label.set_context(CanvasContext(self))
        self._connect_image_label_signals()

        # Font size control. Presets are named entry points into the
        # continuous 8-24pt range; `ui_font_pt` (int) is the single
        # source of truth — see theme.set_font_pt.
        self.font_sizes = {
            "Small": 8,
            "Medium": 10,
            "Large": 12,
            "XL": 14,
            "XXL": 16,
        }  # When adding a new option here, also add it to the Font Size submenu in ui/menu_bar.build_menu_bar.

        # UI prefs persist app-globally via QSettings (not in the .iap
        # project file). Dark mode defaults on — matches the look most
        # users expect from a 2025-era desktop annotation tool; toggle
        # with Settings → Toggle Dark Mode (Ctrl+D).
        self.ui_font_pt, self.dark_mode = load_ui_prefs()

        # Default annotations sorting
        self.current_sort_method = "class"  # Default sorting method

        # DINO batch review state. Initialised eagerly here so the
        # consumers don't each carry a `hasattr` check (one forgotten
        # check would crash with AttributeError).
        self.dino_batch_results: dict[str, list] = {}

        # Setup UI components
        self.setup_ui()

        # Apply theme and font (this includes stylesheet and font size application)
        self.apply_theme_and_font()

        # YOLO Trainer
        self.yolo_trainer = None
        self.setup_yolo_menu()

        # SAM fine-tuning menu + register any previously fine-tuned models so
        # they appear in the SAM model selector (built during setup_ui above).
        self.sam_train_controller.setup_sam_train_menu()
        self.sam_train_controller.refresh_model_selector()

        install_shortcuts(self)
        install_event_filters(self)

        # Start in maximized mode
        self.showMaximized()

    def _connect_image_label_signals(self):
        """Wire ImageLabel events to controller slots. ImageLabel does not
        hold a main_window reference any more — every write path is a
        Qt signal connected here."""
        il = self.image_label
        ac = self.annotation_controller
        cc = self.class_controller
        sc = self.sam_controller

        # Annotation lifecycle
        il.annotationCommitted.connect(ac.add_annotation_to_list)
        il.annotationsBatchSaved.connect(self._on_annotations_batch_saved)
        il.annotationsReplaced.connect(ac.replace_annotations)
        il.annotationListUpdateRequested.connect(ac.update_annotation_list)
        il.annotationSelected.connect(ac.select_annotation_in_list)
        il.canvasSelectionChanged.connect(ac.apply_canvas_selection)
        il.bboxEditCommitted.connect(ac.commit_bbox_edit)
        il.polygonEditCommitted.connect(ac.commit_polygon_edit)
        il.editBaselineRequested.connect(ac.capture_edit_baseline)
        il.deleteSelectionRequested.connect(ac.delete_selected_annotations)
        il.finishPolygonRequested.connect(ac.finish_polygon)
        il.finishRectangleRequested.connect(ac.finish_rectangle)
        il.finishKeypointsRequested.connect(ac.finish_keypoint)
        il.keypointEditCommitted.connect(ac.commit_keypoint_edit)

        # Class
        il.classRequested.connect(cc.add_class)

        # SAM
        il.samPredictionRequested.connect(sc.schedule_sam_prediction)
        il.samPredictionApplyRequested.connect(sc.apply_sam_prediction)
        il.samPredictionAccepted.connect(sc.accept_sam_prediction)
        il.samPointsCleared.connect(sc.cancel_sam_debounce)

        # Tool / UI state
        il.enableToolsRequested.connect(self.enable_tools)
        il.disableToolsRequested.connect(self.disable_tools)
        il.resetToolButtonsRequested.connect(self.reset_tool_buttons)
        il.toolSizeChanged.connect(self._on_tool_size_changed)
        il.selectModeRequested.connect(self._on_select_mode_requested)

        # Navigation / info
        il.zoomInRequested.connect(self.zoom_in)
        il.zoomOutRequested.connect(self.zoom_out)
        il.imageInfoChanged.connect(self.update_image_info)

    def _on_tool_size_changed(self, tool: str, size: int) -> None:
        if tool == "paint":
            self.paint_brush_size = size
        elif tool == "eraser":
            self.eraser_size = size

    def _on_annotations_batch_saved(self) -> None:
        # Push the pre-gesture undo baseline captured at paint-stroke /
        # temp-accept start (no-op if none pending). ADR-026.
        self.annotation_controller.commit_edit_baseline()
        self.annotation_controller.save_current_annotations()
        self.class_controller.update_slice_list_colors()

    def _on_select_mode_requested(self) -> None:
        """Esc on the canvas: deactivate any tool, return to selection mode."""
        self.activate_tool(None)

    def setup_ui(self):
        # Initialize the main layout. tool_group is created inside
        # build_sidebar (it needs to register the tool buttons).
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        build_menu_bar(self)
        build_sidebar(self)
        build_image_area(self)
        build_image_list(self)
        self.setup_slice_list()
        self.update_ui_for_current_tool()

    def update_window_title(self):
        return self.project_controller.update_window_title()

    def new_project(self):
        return self.project_controller.new_project()

    def show_project_search(self):
        from .dialogs.project_search import show_project_search

        show_project_search(self)

    def open_project(self):
        return self.project_controller.open_project()

    def backup_project_before_open(self, project_file):
        return self.project_controller.backup_project_before_open(project_file)

    def restore_project_from_backup(self):
        return self.project_controller.restore_project_from_backup()

    def open_specific_project(self, project_file):
        return self.project_controller.open_specific_project(project_file)

    def load_project_data(self, project_data):
        return self.project_controller.load_project_data(project_data)

    def handle_missing_images(self, missing_images):
        return self.project_controller.handle_missing_images(missing_images)

    def remove_missing_images(self, missing_images):
        return self.project_controller.remove_missing_images(missing_images)

    def prompt_load_missing_images(self, missing_images):
        return self.project_controller.prompt_load_missing_images(missing_images)

    def load_missing_images(self, missing_images):
        return self.project_controller.load_missing_images(missing_images)

    def update_image_list(self):
        return self.image_controller.update_image_list()

    def apply_image_filter(self):
        return self.image_controller.apply_image_filter()

    def select_class(self, index):
        return self.class_controller.select_class(index)

    def close_project(self):
        return self.project_controller.close_project()

    def delete_selected_class(self):
        return self.class_controller.delete_selected_class()

    def check_missing_images(self):
        return self.project_controller.check_missing_images()

    def convert_to_serializable(self, obj):
        return image_utils.convert_to_serializable(obj)

    def save_project(self, show_message=True):
        return self.project_controller.save_project(show_message=show_message)

    def save_project_as(self):
        return self.project_controller.save_project_as()

    def auto_save(self):
        return self.project_controller.auto_save()

    def show_project_details(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        from .dialogs.annotation_statistics import AnnotationStatisticsDialog
        from .dialogs.project_details import ProjectDetailsDialog

        # Generate annotation statistics
        stats_dialog = AnnotationStatisticsDialog(self)
        stats_dialog.generate_statistics(self.all_annotations)

        dialog = ProjectDetailsDialog(self, stats_dialog)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.were_changes_made():
                self.project_notes = dialog.get_notes()
                self.save_project(show_message=False)
                QMessageBox.information(
                    self, "Project Details", "Project details have been updated."
                )
            else:
                print("No changes made to project details.")

    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):
        return self.image_controller.load_multi_slice_image(image_path, dimensions, shape)

    def deactivate_sam_tools(self):
        return self.sam_controller.deactivate_sam_tools()

    def schedule_sam_prediction(self):
        return self.sam_controller.schedule_sam_prediction()

    def apply_sam_prediction(self):
        return self.sam_controller.apply_sam_prediction()

    def accept_sam_prediction(self):
        return self.sam_controller.accept_sam_prediction()

    def setup_slice_list(self):
        return self.image_controller.setup_slice_list()

    def open_images(self):
        return self.image_controller.open_images()

    def convert_to_8bit_rgb(self, image_array):
        return image_utils.convert_to_8bit_rgb(image_array)

    def add_images_to_list(self, file_names):
        return self.image_controller.add_images_to_list(file_names)

    def update_all_images(self, new_image_info):
        return self.image_controller.update_all_images(new_image_info)

    def closeEvent(self, event):
        # check_unsaved_changes prompts and commits/discards as the
        # user chooses; returns False on Cancel.
        if not self.image_label.check_unsaved_changes():
            event.ignore()
            return
        event.accept()

    def switch_slice(self, item):
        return self.image_controller.switch_slice(item)

    def switch_image(self, item):
        return self.image_controller.switch_image(item)

    def adjust_zoom_to_fit(self):
        if not self.current_image:
            return

        # Get the dimensions of the image and the display area
        image_width = self.current_image.width()
        image_height = self.current_image.height()
        display_width = self.scroll_area.viewport().width()
        display_height = self.scroll_area.viewport().height()

        # Calculate and apply the zoom factor to fit the longest side
        zoom_factor = min(display_width / image_width, display_height / image_height)
        self.set_zoom(zoom_factor)

    def activate_current_slice(self):
        return self.image_controller.activate_current_slice()

    def load_image(self, image_path):
        return self.image_controller.load_image(image_path)

    def load_tiff(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
        return self.image_controller.load_tiff(image_path, dimensions, shape, force_dimension_dialog)

    def load_czi(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
        return self.image_controller.load_czi(image_path, dimensions, shape, force_dimension_dialog)

    def load_regular_image(self, image_path):
        return self.image_controller.load_regular_image(image_path)

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        return self.image_controller.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog, axes_hint=axes_hint
        )

    def create_slices(self, image_array, dimensions, image_path):
        return self.image_controller.create_slices(image_array, dimensions, image_path)

    def add_slice_to_list(self, slice_name):
        return self.image_controller.add_slice_to_list(slice_name)

    def normalize_array(self, array):
        return image_utils.normalize_array(array)

    def adjust_contrast(self, image, low_percentile=1, high_percentile=99):
        return image_utils.adjust_contrast(image, low_percentile, high_percentile)

    def activate_slice(self, slice_name):
        return self.image_controller.activate_slice(slice_name)

    def array_to_qimage(self, array):
        return image_utils.array_to_qimage(array)

    def update_slice_list(self):
        return self.image_controller.update_slice_list()

    def clear_slice_list(self):
        return self.image_controller.clear_slice_list()

    def reset_tool_buttons(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)

    def keyPressEvent(self, event):
        # Check if the current focus is on a text editing widget
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        # F2 (Snake game) is wired as a global QShortcut in __init__
        # so it works when child widgets have focus. Don't re-handle here.
        if event.key() == Qt.Key.Key_Delete:
            # Handle deletions
            if self.class_list.hasFocus() and self.class_list.currentItem():
                self.delete_class(self.class_list.currentItem())
            elif (
                self.annotation_list.hasFocus() and self.annotation_list.selectedItems()
            ):
                self.delete_selected_annotations()
            elif self.image_list.hasFocus() and self.image_list.currentItem():
                self.delete_selected_image()
        elif event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Down:
            # Handle slice navigation
            if self.slice_list.hasFocus():
                current_row = self.slice_list.currentRow()
                if event.key() == Qt.Key.Key_Up and current_row > 0:
                    self.slice_list.setCurrentRow(current_row - 1)
                elif (
                    event.key() == Qt.Key.Key_Down
                    and current_row < self.slice_list.count() - 1
                ):
                    self.slice_list.setCurrentRow(current_row + 1)
                self.switch_slice(self.slice_list.currentItem())
            else:
                # Pass the event to the parent for default handling
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Handle accepting visible temporary classes
            if self.has_visible_temp_classes():
                self.accept_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Escape:
            # Handle rejecting visible temporary classes
            if self.has_visible_temp_classes():
                self.reject_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        else:
            # Pass any other key events to the parent for default handling
            super().keyPressEvent(event)

    def has_visible_temp_classes(self):
        return self.dino_controller.has_visible_temp_classes()

    def launch_snake_game(self):
        # print("Launching Snake game")
        if not hasattr(self, "snake_game") or not self.snake_game.isVisible():
            self.snake_game = SnakeGame()
        self.snake_game.show()
        self.snake_game.setFocus()

    def import_annotations(self):
        return io_controller.import_annotations(self)

    def export_annotations(self):
        return io_controller.export_annotations(self)

    def save_slices(self, directory):
        return io_controller.save_slices(self, directory)

    def create_coco_annotation(self, ann, image_id, annotation_id):
        return self.annotation_controller.create_coco_annotation(ann, image_id, annotation_id)

    def update_all_annotation_lists(self):
        return self.annotation_controller.update_all_annotation_lists()

    def update_annotation_list(self, image_name=None):
        return self.annotation_controller.update_annotation_list(image_name)

    def update_slice_list_colors(self):
        return self.class_controller.update_slice_list_colors()

    def update_annotation_list_colors(self, class_name=None, color=None):
        return self.annotation_controller.update_annotation_list_colors(class_name, color)

    def load_image_annotations(self):
        return self.annotation_controller.load_image_annotations()

    def save_current_annotations(self):
        return self.annotation_controller.save_current_annotations()

    def change_font_size(self, size):
        theme.change_font_size(self, size)

    def step_font_size(self, delta):
        theme.step_font_pt(self, delta)

    def reset_font_size(self):
        theme.reset_font_pt(self)

    def unload_ai_models(self):
        """Drop cached SAM/DINO model objects to free GPU/CPU memory.

        Useful on constrained GPUs (e.g. 8 GB) where SAM 2 base + DINO
        base together exhaust VRAM. After unload, the next inference
        call will re-load the model from disk (~1-3 s).
        """
        self.sam_utils.unload()
        self.dino_utils.unload()
        # Reset the dropdowns to a neutral state so the user knows they
        # need to re-pick the model.
        self.sam_model_selector.setCurrentIndex(0)
        if hasattr(self, "dino_model_selector"):
            self.dino_model_selector.setCurrentIndex(0)
            self.dino_model_loaded = False
            self.lbl_dino_status.setText("No DINO model loaded")
            self.btn_detect_single.setEnabled(False)
            self.btn_detect_batch.setEnabled(False)
        QMessageBox.information(
            self,
            "Models Unloaded",
            "SAM and DINO models have been unloaded from memory.\n\n"
            "Note: PyTorch keeps a per-process CUDA context that survives "
            "this unload (typically a few hundred MB visible in Task Manager / "
            "nvidia-smi). To fully reclaim GPU memory, restart the app.\n\n"
            "Re-select a SAM/DINO model to use AI tools again.",
        )

    def toggle_sam_box(self):
        return self.sam_controller.toggle_sam_box()

    def toggle_sam_points(self):
        return self.sam_controller.toggle_sam_points()

    def sort_annotations_by_class(self):
        return self.annotation_controller.sort_annotations_by_class()

    def sort_annotations_by_area(self):
        return self.annotation_controller.sort_annotations_by_area()

    def update_annotation_list_with_sorted(self, sorted_annotations):
        return self.annotation_controller.update_annotation_list_with_sorted(sorted_annotations)

    def change_sam_model(self, model_name):
        return self.sam_controller.change_sam_model(model_name)

    # --- DINO / LLM-Assisted Detection Methods ---

    def _resolve_dino_model_path(self, model_name: str) -> str | None:
        """Return the canonical local path for a preset DINO model, or None if unknown."""
        return self.dino_controller._resolve_dino_model_path(model_name)

    def _on_dino_model_changed(self, text):
        return self.dino_controller._on_dino_model_changed(text)

    def _ensure_dino_model_downloaded(self, model_name):
        return self.dino_controller._ensure_dino_model_downloaded(model_name)

    def browse_dino_model(self):
        return self.dino_controller.browse_dino_model()

    def on_dino_class_row_changed(self):
        return self.dino_controller.on_dino_class_row_changed()

    def _build_dino_class_configs(self):
        return self.dino_controller._build_dino_class_configs()

    def run_dino_detection_single(self):
        return self.dino_controller.run_dino_detection_single()

    def run_dino_detection_batch(self):
        return self.dino_controller.run_dino_detection_batch()


    def _collect_dino_batch_work_items(self):
        return self.dino_controller._collect_dino_batch_work_items()

    def _commit_dino_results(self, image_name, dino_results, sam_results):
        return self.dino_controller._commit_dino_results(image_name, dino_results, sam_results)

    def _store_dino_batch_results(self, image_name, dino_results, sam_results):
        return self.dino_controller._store_dino_batch_results(image_name, dino_results, sam_results)

    def _show_dino_batch_review(self):
        return self.dino_controller._show_dino_batch_review()

    def _navigate_to_image_or_slice(self, name):
        return self.dino_controller._navigate_to_image_or_slice(name)

    def _refresh_dino_temp_for_current(self):
        return self.dino_controller._refresh_dino_temp_for_current()

    def accept_dino_results(self):
        return self.dino_controller.accept_dino_results()

    def reject_dino_results(self):
        return self.dino_controller.reject_dino_results()

    def apply_theme_and_font(self):
        theme.apply_theme_and_font(self)

    def toggle_dark_mode(self):
        theme.toggle_dark_mode(self)

    def apply_stylesheet(self):
        theme.apply_stylesheet(self)

    def update_ui_colors(self):
        theme.update_ui_colors(self)

    ##########    ### Tools  ########## I love useful image processing tools :)
    def open_dataset_splitter(self):
        self.dataset_splitter = DatasetSplitterTool(self)
        self.dataset_splitter.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.dataset_splitter.show_centered(self)

    def show_annotation_statistics(self):
        if not self.all_annotations:
            QMessageBox.warning(
                self, "No Annotations", "There are no annotations to analyze."
            )
            return
        try:
            self.annotation_stats_dialog = show_annotation_statistics(
                self, self.all_annotations
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while showing annotation statistics: {str(e)}",
            )

    def show_coco_json_combiner(self):
        self.coco_json_combiner_dialog = show_coco_json_combiner(self)

    def show_mlflow_settings(self):
        from .dialogs.mlflow_settings_dialog import MLflowSettingsDialog

        MLflowSettingsDialog(self).exec()

    def open_mlflow_ui(self):
        from .training.mlflow_tracker import launch_mlflow_ui, resolve_tracking_uri

        ok, message = launch_mlflow_ui(resolve_tracking_uri(self))
        if ok:
            QMessageBox.information(self, "MLflow UI", message)
        else:
            QMessageBox.warning(self, "MLflow UI", message)

    def show_dino_merge_dialog(self):
        show_dino_merge_dialog(self)

    def show_stack_to_slices(self):
        self.stack_to_slices_dialog = show_stack_to_slices(self)

    def show_image_patcher(self):
        self.image_patcher_dialog = show_image_patcher(self)

    def show_image_augmenter(self):
        self.image_augmenter_dialog = show_image_augmenter(self)

    def show_slice_registration(self):
        self.slice_registration_dialog = SliceRegistrationTool(self)
        self.slice_registration_dialog.show_centered(self)

    def show_stack_interpolator(self):
        self.stack_interpolator_dialog = StackInterpolator(self)
        self.stack_interpolator_dialog.show_centered(self)

    def show_dicom_converter(self):
        self.dicom_converter_dialog = DicomConverter(self)
        self.dicom_converter_dialog.show_centered(self)

    ###################################################################

    # update the show_help method:
    def show_help(self):
        self.help_window = HelpWindow(
            dark_mode=self.dark_mode, font_size=self.ui_font_pt
        )
        self.help_window.show_centered(self)

    def add_images(self):
        if not self.image_label.check_unsaved_changes():
            return
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Add Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)"
        )
        if file_names:
            self.add_images_to_list(file_names)

    def clear_all(self, new_project=False, show_messages=True):
        if not new_project and show_messages:
            reply = self.show_question(
                "Clear All",
                "Are you sure you want to clear all images and annotations? This action cannot be undone.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Clear images
        self.image_list.clear()
        self.image_paths.clear()
        self.all_images.clear()
        self.current_image = None
        self.image_file_name = ""

        # Clear the image display
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())  # Set an empty pixmap
        self.image_label.original_pixmap = None
        self.image_label.scaled_pixmap = None

        # Clear annotations
        self.all_annotations.clear()
        self.annotation_list.setRowCount(0)
        self.image_label.annotations.clear()
        self.image_label.highlighted_annotations.clear()
        self.annotation_controller.clear_history()  # drop undo/redo stacks

        # Clear current class
        self.current_class = None

        # Reset class-related data
        self.class_list.clear()
        self.image_label.class_colors.clear()
        self.class_mapping.clear()
        self.keypoint_schemas.clear()

        # Reset DINO state
        self.dino_class_table.clear_classes()
        self.dino_phrase_panel.clear()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None
        self.dino_model_selector.setCurrentIndex(0)
        self.lbl_dino_status.setText("No DINO model loaded")
        self.btn_detect_single.setEnabled(False)
        self.btn_detect_batch.setEnabled(False)

        # Clear slices
        self.image_slices.clear()
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None
        self.current_stack = None

        # Reset zoom
        self.image_label.zoom_factor = 1.0
        self.zoom_slider.setValue(100)

        # Reset tools
        self.image_label.set_active_tool(None)
        self.polygon_button.setChecked(False)
        self.rectangle_button.setChecked(False)

        # Reset SAM-related attributes
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None

        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)  # Reset cursor to default
        self.sam_model_selector.setCurrentIndex(0)  # Reset to "Pick a SAM Model"
        self.current_sam_model = None  # Reset the current SAM model

        # Reset project-related attributes
        if not new_project:
            if hasattr(self, "current_project_file"):
                del self.current_project_file
            if hasattr(self, "current_project_dir"):
                del self.current_project_dir

        # Update UI
        self.image_label.update()
        self.update_image_info()

        # Force a repaint of the main window
        self.repaint()
        self.update_window_title()

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def update_image_info(self, additional_info=None):
        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            info = f"Image: {width}x{height}"
            if additional_info:
                info += f", {additional_info}"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image loaded")

    def show_question(self, title, message):
        return QMessageBox.question(
            self, title, message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )

    def show_image_context_menu(self, position):
        menu = QMenu()
        current_item = self.image_list.itemAt(position)
        if current_item:
            file_name = current_item.text()
            delete_action = menu.addAction("Remove Image")

            if not self.is_multi_dimensional(file_name):
                predict_action = menu.addAction("Predict using YOLO")

            if self.is_multi_dimensional(file_name):
                redefine_dimensions_action = menu.addAction("Redefine Dimensions")

            action = menu.exec(self.image_list.mapToGlobal(position))

            if action == delete_action:
                self.remove_image()
            elif not self.is_multi_dimensional(file_name) and action == predict_action:
                self.predict_single_image(file_name)
            elif (
                self.is_multi_dimensional(file_name)
                and action == redefine_dimensions_action
            ):
                self.redefine_dimensions(file_name)

    def is_multi_dimensional(self, file_name):
        return self.image_controller.is_multi_dimensional(file_name)

    def predict_single_image(self, file_name):
        return self.yolo_controller.predict_single_image(file_name)

    def redefine_dimensions(self, file_name):
        return self.image_controller.redefine_dimensions(file_name)

    def remove_image(self):
        return self.image_controller.remove_image()

    def load_annotations(self):
        return self.annotation_controller.load_annotations()

    def clear_highlighted_annotation(self):
        return self.annotation_controller.clear_highlighted_annotation()

    def update_highlighted_annotations(self):
        return self.annotation_controller.update_highlighted_annotations()

    def renumber_annotations(self):
        return self.annotation_controller.renumber_annotations()

    def delete_selected_annotations(self):
        return self.annotation_controller.delete_selected_annotations()

    def merge_annotations(self):
        return self.annotation_controller.merge_annotations()

    def delete_selected_image(self):
        return self.image_controller.delete_selected_image()

    def display_image(self):
        return self.image_controller.display_image()

    def update_ui(self):
        self.update_image_list()
        self.update_slice_list()
        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
        self.update_image_info()

    def add_class(self, class_name=None, color=None):
        return self.class_controller.add_class(class_name, color)

    def update_class_item_color(self, item, color):
        return self.class_controller.update_class_item_color(item, color)

    def update_class_list(self):
        return self.class_controller.update_class_list()

    def update_class_selection(self):
        return self.class_controller.update_class_selection()

    def toggle_class_visibility(self, item):
        return self.class_controller.toggle_class_visibility(item)

    def change_annotation_class(self):
        return self.annotation_controller.change_annotation_class()

    def _tool_buttons(self):
        """Map every canvas tool name to its toolbar button. SAM tools share
        the same exclusive set as the manual tools (only one is ever active)."""
        return {
            "polygon": self.polygon_button,
            "rectangle": self.rectangle_button,
            "paint_brush": self.paint_brush_button,
            "eraser": self.eraser_button,
            "keypoint": self.keypoint_button,
            "sam_box": self.sam_box_button,
            "sam_points": self.sam_points_button,
        }

    def activate_tool(self, tool_name):
        """Single choke-point for canvas-tool activation.

        Makes all six tools (manual + SAM) mutually exclusive and keeps
        ``current_tool``, the SAM flags, and the toolbar button checks in
        sync. ``tool_name=None`` returns to selection mode (the canvas
        default). This is why a SAM tool can no longer be active alongside a
        drawing tool, and why Esc can drop back to selection mode.
        """
        il = self.image_label
        is_sam = tool_name in ("sam_box", "sam_points")

        # 1) Exclusive button sync. Block signals so setChecked doesn't
        #    re-enter toggle_tool / toggle_sam_*.
        for name, btn in self._tool_buttons().items():
            btn.blockSignals(True)
            btn.setChecked(name == tool_name)
            btn.blockSignals(False)

        # 2) SAM transient state: clear unless we're entering that SAM tool.
        il.sam_box_active = tool_name == "sam_box"
        il.sam_points_active = tool_name == "sam_points"
        il.sam_bbox = None
        il.sam_positive_points = []
        il.sam_negative_points = []
        if not is_sam:
            self.sam_inference_timer.stop()
            il.drawing_sam_bbox = False
            il.clear_temp_sam_prediction()

        # 3) Switch the active handler (deactivates the previous one). SAM has
        #    no ToolHandler, so this just records current_tool for it.
        il.set_active_tool(tool_name)

        # 4) Cursor + button enable/check refresh.
        il.setCursor(
            Qt.CursorShape.CrossCursor if is_sam else Qt.CursorShape.ArrowCursor
        )
        self.update_ui_for_current_tool()

    def toggle_tool(self):
        if not self.image_label.check_unsaved_changes():
            return

        sender = self.sender()
        if sender is None:
            return

        if not self.current_class:
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Please select a class before using annotation tools.",
            )
            sender.setChecked(False)
            return

        if self.current_class and self.current_class.startswith("Temp-"):
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Cannot use annotation tools with temporary classes.",
            )
            sender.setChecked(False)
            return

        tool_for_button = {
            self.polygon_button: "polygon",
            self.rectangle_button: "rectangle",
            self.paint_brush_button: "paint_brush",
            self.eraser_button: "eraser",
            self.keypoint_button: "keypoint",
        }

        # The keypoint tool needs a pose schema on the current class (#35). Only
        # gate activation — unchecking must always fall through to deactivate, or
        # button state and current_tool would drift on a schemaless class.
        if (
            sender is self.keypoint_button
            and sender.isChecked()
            and self.current_class not in self.keypoint_schemas
        ):
            QMessageBox.warning(
                self,
                "No Keypoint Schema",
                "Define a keypoint schema for this class first "
                "(right-click the class → Define Keypoint Schema).",
            )
            sender.setChecked(False)
            return

        if sender.isChecked():
            self.activate_tool(tool_for_button.get(sender))
            if sender in (self.paint_brush_button, self.eraser_button):
                self.image_label.setFocus()  # paint/eraser need key focus
            elif sender is self.keypoint_button:
                self.image_label.setFocus()  # keypoint needs key focus (Enter/Backspace)
        else:
            self.activate_tool(None)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if self.image_label.current_tool == "paint_brush":
                self.paint_brush_size = max(1, self.paint_brush_size + delta // 120)
                print(f"Paint brush size: {self.paint_brush_size}")
            elif self.image_label.current_tool == "eraser":
                self.eraser_size = max(1, self.eraser_size + delta // 120)
                print(f"Eraser size: {self.eraser_size}")
        else:
            super().wheelEvent(event)

    def update_ui_for_current_tool(self):
        # Disable finish_polygon_button if it still exists in your code
        if hasattr(self, "finish_polygon_button"):
            self.finish_polygon_button.setEnabled(
                self.image_label.current_tool in ["polygon", "rectangle"]
            )

        # Update button states
        self.polygon_button.setChecked(self.image_label.current_tool == "polygon")
        self.rectangle_button.setChecked(self.image_label.current_tool == "rectangle")

        # Disable all tools if no class is selected
        tools_enabled = (
            self.current_class is not None
            and not self.current_class.startswith("Temp-")
        )
        for button in self.tool_group.buttons():
            button.setEnabled(tools_enabled)

        # Update cursor based on the current tool
        if self.image_label.current_tool in ("sam_box", "sam_points"):
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def on_class_selected(self, current=None, previous=None):
        return self.class_controller.on_class_selected(current, previous)

    def disable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)
            button.setEnabled(False)
        self.image_label.set_active_tool(None)

    def enable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setEnabled(True)

    def show_class_context_menu(self, position):
        return self.class_controller.show_class_context_menu(position)

    def change_class_color(self, item):
        return self.class_controller.change_class_color(item)

    def rename_class(self, item):
        return self.class_controller.rename_class(item)

    def delete_class(self, item=None):
        return self.class_controller.delete_class(item)

    def finish_polygon(self):
        return self.annotation_controller.finish_polygon()

    def add_annotation_to_list(self, annotation):
        return self.annotation_controller.add_annotation_to_list(annotation)

    def zoom_in(self):
        new_zoom = min(self.image_label.zoom_factor + 0.1, 5.0)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        new_zoom = max(self.image_label.zoom_factor - 0.1, 0.1)
        self.set_zoom(new_zoom)

    def set_zoom(self, zoom_factor):
        self.image_label.set_zoom(zoom_factor)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.image_label.update()

    def zoom_image(self):
        zoom_factor = self.zoom_slider.value() / 100
        self.set_zoom(zoom_factor)

    def disable_tools(self):
        self.polygon_button.setEnabled(False)
        self.rectangle_button.setEnabled(False)
        # self.finish_polygon_button.setEnabled(False)

    def enable_tools(self):
        self.polygon_button.setEnabled(True)
        self.rectangle_button.setEnabled(True)

    def finish_rectangle(self):
        return self.annotation_controller.finish_rectangle()

    def enter_edit_mode(self, annotation):
        return self.annotation_controller.enter_edit_mode(annotation)

    def exit_edit_mode(self):
        return self.annotation_controller.exit_edit_mode()

    def highlight_annotation_in_list(self, annotation):
        return self.annotation_controller.highlight_annotation_in_list(annotation)

    def select_annotation_in_list(self, annotation):
        return self.annotation_controller.select_annotation_in_list(annotation)

    ################################################################

    def setup_yolo_menu(self):
        return self.yolo_controller.setup_yolo_menu()

    def load_yolo_model(self):
        return self.yolo_controller.load_yolo_model()

    def prepare_yolo_dataset(self):
        return self.yolo_controller.prepare_yolo_dataset()

    def load_yolo_yaml(self):
        return self.yolo_controller.load_yolo_yaml()

    def save_yolo_model(self):
        return self.yolo_controller.save_yolo_model()

    def load_prediction_model(self):
        return self.yolo_controller.load_prediction_model()

    def show_train_dialog(self):
        return self.yolo_controller.show_train_dialog()

    def initialize_yolo_trainer(self):
        return self.yolo_controller.initialize_yolo_trainer()

    def start_training(self, epochs, imgsz):
        return self.yolo_controller.start_training(epochs, imgsz)

    def training_finished(self, results):
        return self.yolo_controller.training_finished(results)

    def set_confidence_threshold(self):
        return self.yolo_controller.set_confidence_threshold()

    def show_predict_dialog(self):
        return self.yolo_controller.show_predict_dialog()

    def run_predictions(self, selected_images):
        return self.yolo_controller.run_predictions(selected_images)

    def process_yolo_results(self, results, image_name):
        return self.yolo_controller.process_yolo_results(results, image_name)

    def add_temp_classes(self, temp_annotations):
        return self.dino_controller.add_temp_classes(temp_annotations)

    def verify_current_class(self):
        return self.dino_controller.verify_current_class()

    def accept_visible_temp_classes(self):
        return self.dino_controller.accept_visible_temp_classes()

    def select_first_primary_class(self):
        return self.dino_controller.select_first_primary_class()

    def reject_visible_temp_classes(self):
        return self.dino_controller.reject_visible_temp_classes()

    def is_class_visible(self, class_name):
        return self.class_controller.is_class_visible(class_name)

    def check_temp_annotations(self):
        return self.dino_controller.check_temp_annotations()

    def remove_all_temp_annotations(self):
        return self.dino_controller.remove_all_temp_annotations()
