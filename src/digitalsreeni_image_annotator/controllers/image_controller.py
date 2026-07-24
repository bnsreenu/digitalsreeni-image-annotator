"""Image / multi-dimensional slice loading and navigation controller.

Extracted from `ImageAnnotator` to give image I/O its own home. Owns:

- Loading from disk (PNG/JPG, TIFF, CZI)
- Multi-dimensional image handling: dimension assignment dialog,
  per-axis slicing, slice list population
- Image / slice navigation (switch_image, switch_slice, activate_slice)
- Display and per-image lifecycle (remove_image, delete_selected_image,
  redefine_dimensions)

State (`current_image`, `current_slice`, `slices`, `image_paths`,
`image_slices`, `image_dimensions`, `image_shapes`, `all_images`,
`image_file_name`, etc.) still lives on the main window and is read here
via `self.mw`. A future phase may migrate ownership of selected
attributes to the controller — for now this is pure method relocation.

The `DimensionDialog` widget lives here too — it is only used by
`process_multidimensional_image`.
"""

import os

from czifile import CziFile
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tifffile import TiffFile

from ..core import image_utils
from ..core.slice_cache import (
    LazySliceList,
    SliceProvider,
    get_shared_lru,
    release_slices,
    slice_names,
)
from ..core.video_handler import (
    VideoHandler,
    VideoSliceProvider,
    file_dialog_filter,
    is_video,
    parse_frame_index,
)

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None, default_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        layout = QVBoxLayout(self)

        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)

        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        self.shape = shape
        dimensions = ["T", "Z", "C", "S", "H", "W"]
        for i, dim in enumerate(shape):
            dim_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            if default_dimensions and i < len(default_dimensions):
                combo.setCurrentText(default_dimensions[i])
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.accept)
        layout.addWidget(self.button)

        self.setMinimumWidth(300)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]


class ImageController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        # Annotation-status badge icons, keyed by (annotated: bool,
        # dark: bool). Each is painted once and reused; cleared on a
        # dark-mode flip via on_theme_changed (issue #43).
        self._status_icon_cache = {}

    def update_image_list(self):
        # Rebuild (and sort) the list, preserving the current selection
        # without switching images.
        self.sort_image_list()

    def sort_image_list(self, select_name=None, do_switch=False):
        """Populate image_list in alphabetical order (upstream issue #60).

        Sorts the model (`all_images`) and the view together so the
        `all_images[i]` ↔ `image_list.item(i)` positional invariant holds
        (relied on by COCO import reconciliation). `setSortingEnabled` is
        deliberately NOT used: `currentRowChanged` is wired to
        `switch_image`, so a live re-sort would fire spurious image
        switches. We rebuild with signals blocked instead, then re-select
        explicitly.

        select_name: file to select after the rebuild (defaults to the
        previously-current item). do_switch: call switch_image once for
        the selected item (used when adding new images).
        """
        current = None
        if self.mw.image_list.currentItem() is not None:
            current = self.mw.image_list.currentItem().text()

        # Grouped images cluster together (ungrouped first, blank group
        # sorts before any name); within a group, by file name (issue #43).
        self.mw.all_images.sort(
            key=lambda info: (
                info.get("group", "").casefold(),
                info.get("file_name", "").casefold(),
                info.get("file_name", ""),
            )
        )

        self.mw.image_list.blockSignals(True)
        self.mw.image_list.clear()
        for info in self.mw.all_images:
            item = QListWidgetItem(info["file_name"])
            group = info.get("group")
            if group:
                # Item TEXT stays the bare file name (many consumers read
                # item(i).text() as a filename — DINO batch nav, COCO
                # import). The group shows only in the tooltip. See #43.
                item.setToolTip(f"{info['file_name']}  [{group}]")
            self.mw.image_list.addItem(item)
        self.mw.image_list.blockSignals(False)

        self._populate_group_combo()

        self.apply_image_filter()

        target = select_name if select_name is not None else current
        if target is not None:
            items = self.mw.image_list.findItems(
                target, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.image_list.blockSignals(True)
                self.mw.image_list.setCurrentItem(items[0])
                self.mw.image_list.blockSignals(False)
                if do_switch:
                    self.switch_image(items[0])

    def image_has_annotations(self, image_info):
        """True if the image (or, for multi-dim images, any of its slices)
        has at least one annotation."""

        def _non_empty(by_class):
            return bool(by_class) and any(by_class.values())

        file_name = image_info["file_name"]
        if _non_empty(self.mw.all_annotations.get(file_name, {})):
            return True

        if image_info.get("is_multi_slice", False):
            base_name = os.path.splitext(file_name)[0]
            slices = self.mw.image_slices.get(base_name)
            if slices:
                # Name-only scan (slice_names) — never materialise QImages
                # just to check annotation status (issue #45).
                return any(
                    _non_empty(self.mw.all_annotations.get(slice_name, {}))
                    for slice_name in slice_names(slices)
                )
            # Slices not extracted yet (e.g. load cancelled) — slice keys
            # are f"{base_name}_T1_Z5_..." so a "{base_name}_" prefix match
            # is exact enough; a bare substring match would not be. Caveat:
            # this also matches "{base_name}_8bit" artifact keys, which
            # redefine_dimensions deliberately excludes — acceptable here
            # since an _8bit key with annotations still means "this image
            # has annotations".
            prefix = base_name + "_"
            return any(
                key.startswith(prefix) and _non_empty(by_class)
                for key, by_class in self.mw.all_annotations.items()
            )

        return False

    def apply_image_filter(self):
        """Hide image-list rows that don't match the annotation-status
        filter (upstream issue #27).

        Rows are hidden via setRowHidden, never removed: other code
        (DINO batch navigation, COCO import) iterates the list by row
        index, and hiding fires no currentRowChanged so it cannot
        trigger a spurious switch_image.

        A non-matching row is hidden even when it is the current
        selection — hiding does not change `current_image`, so the canvas
        keeps showing the worked-on image while its row leaves the list
        (e.g. the image just gained its first annotation under the
        "Without annotations" filter). Keyboard nav skips hidden rows.
        """
        combo = getattr(self.mw, "image_filter_combo", None)
        if combo is None:
            return
        mode = combo.currentIndex()  # 0 = all, 1 = without, 2 = with

        # Group filter (issue #43): a specific group selected (index > 0)
        # hides rows whose image isn't in it. Index 0 ("All groups") means
        # "hide nothing", so the default path below stays a cheap unhide.
        group_combo = getattr(self.mw, "image_group_combo", None)
        active_group = None
        if group_combo is not None and group_combo.currentIndex() > 0:
            active_group = group_combo.currentText()

        if mode == 0 and active_group is None:
            # Default case runs on every update_slice_list_colors — skip the
            # per-row hide computation (nothing is filtered) but still refresh
            # the status badges, which necessarily scan annotation state per
            # row (unavoidable: badges must stay current after every mutation).
            for i in range(self.mw.image_list.count()):
                self.mw.image_list.setRowHidden(i, False)
            self.refresh_image_status_icons()
            return
        infos = {info["file_name"]: info for info in self.mw.all_images}
        for i in range(self.mw.image_list.count()):
            info = infos.get(self.mw.image_list.item(i).text())
            if mode == 0:
                status_hide = False
            else:
                annotated = bool(info) and self.image_has_annotations(info)
                status_hide = annotated if mode == 1 else not annotated
            if active_group is None:
                group_hide = False
            else:
                group_hide = not (info and info.get("group") == active_group)
            # Hide the row if EITHER filter excludes it.
            self.mw.image_list.setRowHidden(i, status_hide or group_hide)

        self.refresh_image_status_icons()

    def refresh_image_status_icons(self):
        """Set a per-row annotation-status badge on the image list (#43).

        Filled green dot = the image (or, for a multi-dim stack, any of
        its slices) has annotations; hollow gray dot = none. Both states
        are derived — nothing is stored. Icons are cached per (annotated,
        dark_mode) and painted once; the colours are theme-tuned (brighter
        on the dark sidebar), so on_theme_changed clears the cache on a
        dark-mode flip to force a repaint at the new theme's colours.

        Called at the end of apply_image_filter (so it refreshes on every
        annotation mutation via update_slice_list_colors) and after
        sort_image_list's rebuild.
        """
        dark = bool(getattr(self.mw, "dark_mode", False))
        infos = {info["file_name"]: info for info in self.mw.all_images}
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            info = infos.get(item.text())
            annotated = bool(info) and self.image_has_annotations(info)
            item.setIcon(self._status_icon(annotated, dark))

    def _status_icon(self, annotated, dark):
        key = (annotated, dark)
        icon = self._status_icon_cache.get(key)
        if icon is None:
            icon = self._build_status_icon(annotated, dark)
            self._status_icon_cache[key] = icon
        return icon

    @staticmethod
    def _build_status_icon(annotated, dark):
        """Paint a 12x12 status dot into a QIcon, tuned for the theme.

        These are painted PIXMAPS, not stylesheet colours, so the "No
        Hardcoded Colors Rule" (which targets setStyleSheet literals) does
        not apply. The colours are picked per theme: a brighter green /
        lighter gray on the soft-dark sidebar for adequate contrast, a
        deeper pair on the light one. That difference is what makes the
        (annotated, dark) cache dimension and on_theme_changed do real work.
        """
        if annotated:
            fill = QColor(63, 185, 80) if dark else QColor(46, 160, 67)
        else:
            outline = QColor(155, 155, 155) if dark else QColor(120, 120, 120)
        pixmap = QPixmap(12, 12)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if annotated:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)  # filled dot
            painter.drawEllipse(2, 2, 8, 8)
        else:
            pen = QPen(outline)  # hollow outline dot
            pen.setWidthF(1.5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(3, 3, 6, 6)
        painter.end()
        return QIcon(pixmap)

    def on_theme_changed(self):
        """Rebuild the status-icon cache after a dark-mode flip (#43).

        Called from the theme choke point (ui/theme.toggle_dark_mode).
        The cached icons are keyed by dark-mode, so clearing forces a
        redraw at the new theme on the next refresh.
        """
        self._status_icon_cache.clear()
        self.refresh_image_status_icons()

    def set_image_group(self, file_name, group):
        """Assign or clear an image's group tag (issue #43).

        group: a name (whitespace stripped) or None/empty to remove the
        image from any group. Re-sorts the list so grouped images cluster,
        then auto-saves — but never during project load (CLAUDE.md guard,
        matching add_images_to_list).
        """
        info = next(
            (img for img in self.mw.all_images if img["file_name"] == file_name),
            None,
        )
        if info is None:
            return
        normalized = group.strip() if isinstance(group, str) else None
        normalized = normalized or None
        # No-op if nothing actually changes (e.g. "Remove from group" on an
        # already-ungrouped image) so we don't re-sort + auto_save for nothing.
        if normalized == info.get("group"):
            return
        if normalized:
            info["group"] = normalized
        else:
            info.pop("group", None)
        self.sort_image_list()
        if not self.mw.is_loading_project:
            self.mw.auto_save()

    def _populate_group_combo(self):
        """Repopulate image_group_combo from the derived group set (#43).

        Signals are blocked (repopulating fires currentIndexChanged →
        apply_image_filter otherwise) and the current selection is
        preserved by text, falling back to "All groups" if its group is
        gone.
        """
        combo = getattr(self.mw, "image_group_combo", None)
        if combo is None:
            return
        groups = sorted(
            {info.get("group") for info in self.mw.all_images if info.get("group")}
        )
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All groups")
        for g in groups:
            combo.addItem(g)
        idx = combo.findText(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def setup_slice_list(self):
        self.mw.slice_list = QListWidget()
        self.mw.slice_list.itemClicked.connect(self.switch_slice)
        self.mw.image_list_layout.addWidget(QLabel("Slices:"))
        self.mw.image_list_layout.addWidget(self.mw.slice_list)

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self.mw, "Open Images or Videos", "", file_dialog_filter()
        )
        if file_names:
            self.mw.image_list.clear()
            self.mw.image_paths.clear()
            self.mw.all_images.clear()
            self.mw.slice_list.clear()
            # Drop the outgoing stacks' cached QImages AND their retained
            # source arrays: image_slices is being replaced wholesale, so wipe
            # the shared slice LRU and clear image_slices. Under Strategy A a
            # LazySliceList pins its whole decoded ndarray, so merely
            # rebinding mw.slices = [] (mw.slices is no longer the same object
            # as image_slices[base]) would leak every previously-open stack for
            # the session. Mirrors clear_all (issue #45).
            get_shared_lru().clear()
            self.mw.image_slices.clear()
            # Video handlers own an open cv2.VideoCapture each — release them
            # before the dict is wiped so no file handle leaks (issue #47).
            for handler in self.mw.video_handlers.values():
                handler.release()
            self.mw.video_handlers.clear()
            self.mw.slices = []
            self.mw.current_stack = None
            self.mw.current_slice = None
            self.add_images_to_list(file_names)

    def add_images_to_list(self, file_names):
        first_added_name = None
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            if base_name not in self.mw.image_paths:
                # Slices, video frames and their annotations are all keyed by
                # the EXT-STRIPPED base name, so `video.mp4` and `video.tif`
                # would silently clobber each other's slices. Refuse the
                # colliding file instead (issue #47).
                if self._base_name_collision(base_name):
                    QMessageBox.warning(
                        self.mw,
                        "Duplicate Name",
                        f"An image named "
                        f"'{os.path.splitext(base_name)[0]}' is already loaded "
                        "from a different file. Rename one of the files and "
                        "reopen it — slices and annotations are keyed by this "
                        "base name.",
                    )
                    continue

                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.mw.all_images) + 1,
                    "is_multi_slice": False,
                }

                if file_name.lower().endswith((".tif", ".tiff", ".czi")):
                    try:
                        self.load_multi_slice_image(file_name)
                    except ValueError as e:
                        # LZW/compressed TIFFs need the optional imagecodecs
                        # package; without it tifffile raises ValueError and
                        # the app used to crash (#56). Skip the file with an
                        # actionable message instead of a half-added entry.
                        if self._is_missing_codec_error(e):
                            QMessageBox.critical(
                                self.mw,
                                "Cannot open TIFF",
                                f"'{base_name}' uses a compression that requires "
                                "the 'imagecodecs' package, which is not "
                                "installed.\n\nInstall it with:\n"
                                "    pip install imagecodecs\n\n"
                                "then reopen the image.",
                            )
                            continue
                        raise
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    if (
                        base_name_without_ext in self.mw.image_slices
                        and self.mw.image_slices[base_name_without_ext]
                    ):
                        first_slice_name, first_slice = self.mw.image_slices[
                            base_name_without_ext
                        ][0]
                        image_info["height"] = first_slice.height()
                        image_info["width"] = first_slice.width()
                        image_info["is_multi_slice"] = True
                        image_info["dimensions"] = self.mw.image_dimensions.get(
                            base_name_without_ext, []
                        )
                        image_info["shape"] = self.mw.image_shapes.get(
                            base_name_without_ext, []
                        )
                elif is_video(file_name):
                    # Video frames become lazy slices (issue #47). Read the
                    # probe from the handler's metadata — do NOT force a frame
                    # decode just to fill height/width. `dimensions`/`shape`
                    # are omitted (a video has none).
                    self.load_video(file_name)
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    handler = self.mw.video_handlers[base_name_without_ext]
                    image_info["height"] = handler.height
                    image_info["width"] = handler.width
                    image_info["is_multi_slice"] = True
                    image_info["is_video"] = True
                    image_info["video_metadata"] = handler.metadata()
                else:
                    image = QImage(file_name)
                    image_info["height"] = image.height()
                    image_info["width"] = image.width()

                self.mw.all_images.append(image_info)
                if first_added_name is None:
                    first_added_name = base_name

                self.mw.image_paths[base_name] = file_name

        # Rebuild the list in sorted order and select/switch to the first
        # newly added image. Skipped during project load (the list is
        # rebuilt once via update_ui afterwards, and load picks row 0) to
        # avoid an O(n^2) re-sort per image.
        if first_added_name is not None and not self.mw.is_loading_project:
            self.sort_image_list(select_name=first_added_name, do_switch=True)
        else:
            self.apply_image_filter()

        if not self.mw.is_loading_project:
            self.mw.auto_save()

    @staticmethod
    def _is_missing_codec_error(exc):
        """True if a tifffile read failed because the imagecodecs package
        is unavailable for the TIFF's compression — e.g. LZW (#56).

        Matches only the reliable 'imagecodecs' token: tifffile names the
        package in every such message. A broader 'compression' match would
        silently swallow unrelated ValueErrors behind a misleading dialog.
        """
        return "imagecodecs" in str(exc).lower()

    def _base_name_collision(self, base_name):
        """True if a DIFFERENT already-loaded image shares this file's
        ext-stripped base name (issue #47).

        Slice keys, video frame keys and ``image_slices`` are all keyed by
        the ext-stripped base, so two files that differ only by extension
        (``video.mp4`` vs ``video.tif``) would clobber each other's slices.
        """
        base_no_ext = os.path.splitext(base_name)[0]
        for info in self.mw.all_images:
            other = info["file_name"]
            if other != base_name and os.path.splitext(other)[0] == base_no_ext:
                return True
        return False

    def load_video(self, image_path):
        """Load a video as a stack whose slices are its frames (issue #47).

        Reuses the #45 lazy-slice machinery: a :class:`VideoSliceProvider`
        decodes each frame on demand and the shared ``SliceLRU`` bounds the
        live QImages. ``mw.slices`` and ``mw.image_slices[base]`` are the SAME
        :class:`LazySliceList` object (the #45 guardrail), so every existing
        slice consumer works for video unchanged.
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Release + replace any prior handler / slice list under this base
        # BEFORE overwriting, mirroring create_slices' release-before-replace:
        # the old objects stay referenced until the reassignment below, so the
        # new provider is guaranteed a distinct id() and a recycled id can't
        # alias stale LRU entries.
        old_handler = self.mw.video_handlers.get(base_name)
        if old_handler is not None:
            old_handler.release()
        release_slices(self.mw.image_slices.get(base_name))

        handler = VideoHandler(image_path)
        self.mw.video_handlers[base_name] = handler

        provider = VideoSliceProvider(handler, base_name)
        lazy = LazySliceList(provider)
        self.mw.image_slices[base_name] = lazy
        self.mw.slices = lazy

        self.mw.slice_list.clear()
        for slice_name in lazy.names:
            self.add_slice_to_list(slice_name)

        if lazy:
            first_name, first_image = lazy[0]
            self.mw.current_image = first_image
            self.mw.current_slice = first_name
            self.mw.slice_list.setCurrentRow(0)
            self.activate_slice(self.mw.current_slice)

            meta = handler.metadata()
            slice_info = (
                f"Total frames: {meta['total_frames']}, "
                f"{meta['width']}x{meta['height']}, {meta['fps']:.2f} fps"
            )
            self.mw.update_image_info(additional_info=slice_info)
        else:
            logger.warning(f"No frames decoded for video: {base_name}")

        logger.info(f"Loaded video {base_name}: {handler.total_frames} frames")
        return lazy

    def annotated_frame_indices(self, base_name):
        """Frame indices of ``base_name`` (a video) that carry annotations.

        A frame is annotated if ``all_annotations[frame_key]`` is non-empty and
        holds at least one non-empty class list. Prefix-match ``base + "_F"``
        and parse the exact index (a bare ``base + "_"`` would also swallow
        multi-dim slice keys) — see issue #48.
        """
        prefix = base_name + "_F"
        out = set()
        for key, by_class in self.mw.all_annotations.items():
            if key.startswith(prefix) and by_class and any(by_class.values()):
                idx = parse_frame_index(key)
                if idx is not None:
                    out.add(idx)
        return out

    def current_video(self):
        """``(base_name, handler, info)`` if the active image is a loaded video,
        else ``None`` (issue #48).

        Single source of truth for the ``currentItem → is_video → handler``
        resolution shared by the timeline sync, Home/End nav and frame export,
        so those three call sites can't drift.
        """
        item = self.mw.image_list.currentItem()
        if item is None:
            return None
        info = next(
            (img for img in self.mw.all_images if img["file_name"] == item.text()),
            None,
        )
        if not (info and info.get("is_video")):
            return None
        base_name = os.path.splitext(info["file_name"])[0]
        handler = self.mw.video_handlers.get(base_name)
        if handler is None:
            return None
        return base_name, handler, info

    def update_video_timeline(self):
        """Sync the video timeline widget to the active image (issue #48).

        Shows + configures the timeline when the active image is a video,
        otherwise hides it. The timeline is a pure VIEW: it never changes the
        frame itself (user scrubs route back through ``switch_slice`` via
        ``on_timeline_frame_selected``). Called after every frame switch and at
        the end of ``update_slice_list_colors`` so annotation marks stay current
        without a frame change.
        """
        timeline = getattr(self.mw, "video_timeline", None)
        if timeline is None:
            return

        video = self.current_video()
        if video is not None:
            base_name, handler, _info = video
            timeline.set_video(handler.total_frames, handler.fps)
            idx = parse_frame_index(self.mw.current_slice or "")
            if idx is not None:
                timeline.set_current_frame(idx)
            timeline.set_frame_states(self.video_frame_states(base_name))
            timeline.setVisible(True)
        else:
            timeline.setVisible(False)

    def video_frame_states(self, base_name):
        """Per-frame timeline states for the video ``base_name`` (issue #51).

        Returns ``{frame_idx: state}`` where ``state`` is one of
        ``"tracked"`` / ``"needs_review"`` / ``"annotated"``. Cheap name-only
        scans of ``all_annotations`` + ``dino_batch_results`` (no frame decode).

        Precedence (highest wins, so a frame appears once at its top state):
        ``needs_review`` (a pending SAM 3 / DINO review result for this frame)
        > ``tracked`` (holds a committed ``source == "sam3-track"`` annotation)
        > ``annotated`` (any other committed annotation). ``needs_review`` is
        applied last so it overrides a committed state on the same frame.
        """
        prefix = base_name + "_F"
        states = {}
        for key, by_class in self.mw.all_annotations.items():
            if not key.startswith(prefix):
                continue
            idx = parse_frame_index(key)
            if idx is None:
                continue
            anns = [a for lst in by_class.values() for a in lst]
            if not anns:
                continue
            if any(a.get("source") == "sam3-track" for a in anns):
                states[idx] = "tracked"
            else:
                states[idx] = "annotated"
        # Pending review results override any committed state on the frame.
        for key in getattr(self.mw, "dino_batch_results", {}):
            if not key.startswith(prefix):
                continue
            idx = parse_frame_index(key)
            if idx is not None:
                states[idx] = "needs_review"
        return states

    def update_all_images(self, new_image_info):
        for info in new_image_info:
            if not any(
                img["file_name"] == info["file_name"] for img in self.mw.all_images
            ):
                self.mw.all_images.append(info)

    def switch_slice(self, item):
        if item is None:
            return
        # check_unsaved_changes prompts the user and commits/discards
        # all dirty tool handlers; returns False on Cancel.
        if not self.mw.image_label.check_unsaved_changes():
            return

        self.mw.save_current_annotations()
        self.mw.image_label.clear_temp_sam_prediction()
        # Exit vertex-edit mode (as switch_image does) so editing_polygon /
        # _editing_polygon_orig don't linger onto the next slice (ADR-026).
        self.mw.image_label.exit_editing_mode()
        self.mw.annotation_controller.reset_coalesce()

        slice_name = item.text()
        # Lazy materialise this slice (LRU-cached) instead of scanning +
        # holding every slice's QImage; prefetch neighbours so Up/Down nav
        # stays instant (issue #45).
        qimage = self.mw.slices.get(slice_name)
        if qimage is not None:
            self.mw.slices.prefetch_around(slice_name)
            self.mw.current_image = qimage
            self.mw.current_slice = slice_name
            self.display_image()
            self.mw.load_image_annotations()
            self.mw.update_annotation_list()
            self.mw.clear_highlighted_annotation()
            self.mw.image_label.reset_annotation_state()
            self.mw.image_label.clear_current_annotation()
            self.mw.update_image_info()

        self.mw.image_label.update()
        self.mw.update_slice_list_colors()

        self.mw.set_zoom(1.0)

        self.mw._refresh_dino_temp_for_current()

    def switch_image(self, item):
        if item is None:
            return
        if not self.mw.image_label.check_unsaved_changes():
            return

        current_item = self.mw.image_list.currentItem()

        if not self.mw.check_temp_annotations():
            self.mw.image_list.setCurrentItem(current_item)
            return

        self.mw.save_current_annotations()
        self.mw.image_label.clear_temp_sam_prediction()
        self.mw.image_label.exit_editing_mode()
        self.mw.annotation_controller.reset_coalesce()

        file_name = item.text()
        logger.debug(f"Switching to image: {file_name}")

        image_info = next(
            (img for img in self.mw.all_images if img["file_name"] == file_name), None
        )

        if image_info:
            self.mw.image_file_name = file_name
            image_path = self.mw.image_paths.get(file_name)

            if not image_path:
                image_path = os.path.join(
                    self.mw.current_project_dir, "images", file_name
                )

            if image_path and os.path.exists(image_path):
                if image_info.get("is_multi_slice", False):
                    base_name = os.path.splitext(file_name)[0]
                    if base_name in self.mw.image_slices:
                        self.mw.slices = self.mw.image_slices[base_name]
                        if self.mw.slices:
                            self.mw.current_image = self.mw.slices[0][1]
                            self.mw.current_slice = self.mw.slices[0][0]
                            self.update_slice_list()
                            self.activate_slice(self.mw.current_slice)
                    elif image_info.get("is_video"):
                        # Video whose frames aren't in image_slices yet (e.g. a
                        # future path that dropped them): rebuild via load_video,
                        # never load_multi_slice_image (which handles neither
                        # branch for a video and would leave a stale display).
                        self.load_video(image_path)
                    else:
                        self.load_multi_slice_image(
                            image_path,
                            image_info.get("dimensions"),
                            image_info.get("shape"),
                        )
                else:
                    self.load_regular_image(image_path)
                    self.display_image()
                    self.clear_slice_list()

                self.mw.load_image_annotations()
                self.mw.update_annotation_list()
                self.mw.clear_highlighted_annotation()
                self.mw.image_label.update()
                self.mw.image_label.reset_annotation_state()
                self.mw.image_label.clear_current_annotation()
                self.mw.update_image_info()

                self.mw.adjust_zoom_to_fit()
            else:
                self.mw.current_image = None
                self.mw.image_label.clear()
                self.mw.load_image_annotations()
                self.mw.update_annotation_list()
                self.mw.update_image_info()

            self.mw.image_list.setCurrentItem(item)
            self.mw.image_label.update()
            self.mw.update_slice_list_colors()
        else:
            self.mw.current_image = None
            self.mw.current_slice = None
            self.mw.image_label.clear()
            self.mw.update_image_info()
            self.clear_slice_list()

        self.mw._refresh_dino_temp_for_current()

    def activate_current_slice(self):
        if self.mw.current_slice:
            items = self.mw.slice_list.findItems(
                self.mw.current_slice, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.slice_list.setCurrentItem(items[0])

            self.mw.load_image_annotations()
            self.mw.image_label.update()
            self.mw.update_annotation_list()

    def load_image(self, image_path):
        extension = os.path.splitext(image_path)[1].lower()
        if extension in [".tif", ".tiff"]:
            self.load_tiff(image_path)
        elif extension == ".czi":
            self.load_czi(image_path)
        else:
            self.load_regular_image(image_path)

    def load_tiff(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        logger.debug(f"Loading TIFF file: {image_path}")
        axes_hint = None
        with TiffFile(image_path) as tif:
            logger.debug(f"TIFF tags: {tif.pages[0].tags}")

            try:
                metadata = tif.pages[0].tags["ImageDescription"].value
                logger.debug(f"TIFF metadata: {metadata}")
            except KeyError:
                logger.debug("No ImageDescription metadata found")

            try:
                series_axes = tif.series[0].axes if tif.series else None
                if series_axes:
                    axis_map = {
                        "T": "T", "Z": "Z", "C": "C", "S": "S",
                        "Y": "H", "X": "W",
                    }
                    mapped = [axis_map.get(a) for a in series_axes]
                    if all(a is not None for a in mapped):
                        axes_hint = mapped
                        logger.debug(f"TIFF series axes: {series_axes} → dimension hint: {axes_hint}")
                    else:
                        unknown = [a for a in series_axes if axis_map.get(a) is None]
                        logger.debug(f"TIFF series axes had unknown labels {unknown}, no hint applied")
            except Exception:
                logger.exception("Could not read TIFF series axes")

            if len(tif.pages) > 1:
                logger.debug(f"Multi-page TIFF detected. Number of pages: {len(tif.pages)}")
                image_array = tif.asarray()
            else:
                logger.debug("Single-page TIFF detected.")
                image_array = tif.pages[0].asarray()

            logger.debug(f"Image array shape: {image_array.shape}")
            logger.debug(f"Image array dtype: {image_array.dtype}")
            logger.debug(f"Image min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            logger.debug("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog,
            axes_hint=axes_hint,
        )

    def load_czi(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        logger.debug(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            logger.debug(f"CZI array shape: {image_array.shape}")
            logger.debug(f"CZI array dtype: {image_array.dtype}")
            logger.debug(f"CZI array min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            logger.debug("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog
        )

    def load_regular_image(self, image_path):
        self.mw.current_image = QImage(image_path)
        self.mw.slices = []
        self.mw.slice_list.clear()
        self.mw.current_slice = None

    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        logger.debug(f"Loading multi-slice image: {image_path}")
        logger.debug(f"Base name: {base_name}")

        if dimensions and shape:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            self.mw.image_dimensions[base_name] = dimensions
            self.mw.image_shapes[base_name] = shape
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path, dimensions, shape)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path, dimensions, shape)
        else:
            logger.debug("No stored dimensions or shape, loading as new image")
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path)

        logger.debug(f"Loaded multi-slice image: {file_name}")
        logger.debug(f"Dimensions: {self.mw.image_dimensions.get(base_name, 'Not found')}")
        logger.debug(f"Shape: {self.mw.image_shapes.get(base_name, 'Not found')}")
        logger.debug(f"Number of slices: {len(self.mw.slices)}")

        if self.mw.slices:
            self.mw.current_image = self.mw.slices[0][1]
            self.mw.current_slice = self.mw.slices[0][0]

            self.update_slice_list()
            self.mw.slice_list.setCurrentRow(0)
            self.activate_slice(self.mw.current_slice)
            logger.debug(f"Activated first slice: {self.mw.current_slice}")
        else:
            logger.warning("No slices were loaded")
            self.mw.current_image = None
            self.mw.current_slice = None

        self.update_slice_list()
        self.mw.image_label.update()

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        logger.debug(f"Processing file: {file_name}")
        logger.debug(f"Image array shape: {image_array.shape}")
        logger.debug(f"Image array dtype: {image_array.dtype}")

        if dimensions is None or force_dimension_dialog:
            if image_array.ndim > 2:
                # ndim≥5 had a `[-ndim:]` slice bug that produced 2560 wrong
                # slices on a 5D TZCYX file — see arc42.
                if axes_hint and len(axes_hint) == image_array.ndim:
                    default_dimensions = list(axes_hint)
                    logger.debug(f"Applying axes hint as default dims: {default_dimensions}")
                else:
                    if axes_hint and len(axes_hint) != image_array.ndim:
                        logger.debug(
                            f"Ignoring axes hint (length {len(axes_hint)} "
                            f"vs ndim {image_array.ndim})"
                        )
                    ndim_defaults = {
                        3: ["Z", "H", "W"],
                        4: ["T", "Z", "H", "W"],
                        5: ["T", "Z", "C", "H", "W"],
                        6: ["T", "Z", "C", "S", "H", "W"],
                    }
                    default_dimensions = ndim_defaults.get(
                        image_array.ndim,
                        ["T"] * max(0, image_array.ndim - 2) + ["H", "W"],
                    )

                progress = QProgressDialog(
                    "Assigning dimensions...", "Cancel", 0, 100, self.mw
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(10)
                QApplication.processEvents()

                while True:
                    dialog = DimensionDialog(
                        image_array.shape, file_name, self.mw, default_dimensions
                    )
                    progress.setValue(50)
                    QApplication.processEvents()
                    if dialog.exec():
                        dimensions = dialog.get_dimensions()
                        logger.debug(f"Assigned dimensions: {dimensions}")
                        if "H" in dimensions and "W" in dimensions:
                            self.mw.image_dimensions[base_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(
                                self.mw,
                                "Invalid Dimensions",
                                "You must assign both H and W dimensions.",
                            )
                    else:
                        progress.close()
                        return
                progress.setValue(100)
                progress.close()
            else:
                dimensions = ["H", "W"]
                self.mw.image_dimensions[base_name] = dimensions

        self.mw.image_shapes[base_name] = image_array.shape
        logger.debug(f"Final assigned dimensions: {self.mw.image_dimensions[base_name]}")
        logger.debug(f"Image shape: {self.mw.image_shapes[base_name]}")

        if self.mw.image_dimensions[base_name]:
            self.create_slices(
                image_array, self.mw.image_dimensions[base_name], image_path
            )
        else:
            rgb_image = image_utils.convert_to_8bit_rgb(image_array)
            self.mw.current_image = image_utils.array_to_qimage(rgb_image)
            self.mw.slices = []
            self.mw.slice_list.clear()

        if self.mw.slices:
            self.mw.current_image = self.mw.slices[0][1]
            self.mw.current_slice = self.mw.slices[0][0]
            self.mw.slice_list.setCurrentRow(0)
            self.mw.load_image_annotations()
            self.mw.image_label.update()

        self.mw.update_image_info()

        self.update_slice_list()
        self.mw.update_annotation_list()
        self.mw.image_label.update()

    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.mw.slice_list.clear()

        logger.debug(f"Creating slices for {base_name}")
        logger.debug(f"Dimensions: {dimensions}")
        logger.debug(f"Image array shape: {image_array.shape}")

        # Reloading a stack under the same base_name (redefine dims, project
        # reload) drops the previous LazySliceList — evict its cached QImages
        # first so a recycled provider id() can't alias stale entries. The old
        # object stays referenced until the reassignment below, so the new
        # provider is guaranteed a distinct id() (issue #45).
        release_slices(self.mw.image_slices.get(base_name))

        # Lazy slicing (issue #45): retain the already-decoded source array in
        # a provider and materialise each slice's QImage ON DEMAND through a
        # shared bounded LRU. No per-slice pixel work happens here now, so the
        # progress dialog (pixel work only) is gone; building names is cheap.
        provider = SliceProvider(image_array, dimensions, base_name)
        lazy = LazySliceList(provider)

        for slice_name in lazy.names:
            self.add_slice_to_list(slice_name)

        # mw.image_slices[base] and mw.slices MUST be the same object — several
        # paths compare/assign them (issue #45 guardrail).
        self.mw.image_slices[base_name] = lazy
        self.mw.slices = lazy

        if lazy:
            first_name, first_image = lazy[0]
            self.mw.current_image = first_image
            self.mw.current_slice = first_name
            self.mw.slice_list.setCurrentRow(0)

            self.activate_slice(self.mw.current_slice)

            slice_info = f"Total slices: {len(lazy)}"
            for dim, size in zip(dimensions, image_array.shape):
                if dim not in ["H", "W"]:
                    slice_info += f", {dim}: {size}"
            self.mw.update_image_info(additional_info=slice_info)
        else:
            logger.warning("No slices were created")

        logger.info(f"Created {len(lazy)} slices for {base_name}")
        return lazy

    def add_slice_to_list(self, slice_name):
        item = QListWidgetItem(slice_name)

        if self.mw.dark_mode:
            item.setBackground(QColor(40, 40, 40))
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(235, 235, 235))
                item.setBackground(QColor(58, 95, 140))
            else:
                item.setForeground(QColor(200, 200, 200))
        else:
            item.setBackground(QColor(240, 240, 240))
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(255, 255, 255))
                item.setBackground(QColor(70, 130, 180))
            else:
                item.setForeground(QColor(0, 0, 0))

        self.mw.slice_list.addItem(item)

    def activate_slice(self, slice_name):
        self.mw.current_slice = slice_name
        self.mw.image_file_name = slice_name
        self.mw.load_image_annotations()
        self.mw.update_annotation_list()

        # Lazy materialise (LRU-cached) + prefetch neighbours (issue #45).
        qimage = self.mw.slices.get(slice_name)
        if qimage is not None:
            self.mw.slices.prefetch_around(slice_name)
            self.mw.current_image = qimage
            self.display_image()

        self.mw.image_label.update()

        items = self.mw.slice_list.findItems(slice_name, Qt.MatchFlag.MatchExactly)
        if items:
            self.mw.slice_list.setCurrentItem(items[0])

    def update_slice_list(self):
        self.mw.slice_list.clear()
        # Name-only (slice_names) — rebuilding the list must not decode pixels
        # (issue #45).
        for slice_name in slice_names(self.mw.slices):
            item = QListWidgetItem(slice_name)
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(Qt.GlobalColor.green))
            else:
                item.setForeground(
                    QColor(Qt.GlobalColor.black)
                    if not self.mw.dark_mode
                    else QColor(Qt.GlobalColor.white)
                )
            self.mw.slice_list.addItem(item)

        if self.mw.current_slice:
            items = self.mw.slice_list.findItems(
                self.mw.current_slice, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.slice_list.setCurrentItem(items[0])

    def clear_slice_list(self):
        self.mw.slice_list.clear()
        self.mw.slices = []
        self.mw.current_slice = None

    def is_multi_dimensional(self, file_name):
        return file_name.lower().endswith((".tif", ".tiff", ".czi"))

    def redefine_dimensions(self, file_name):
        file_path = self.mw.image_paths.get(file_name)
        if not file_path or not file_path.lower().endswith((".tif", ".tiff", ".czi")):
            return

        reply = QMessageBox.warning(
            self.mw,
            "Redefine Dimensions",
            "Redefining dimensions will cause all associated annotations to be lost. "
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            base_name = os.path.splitext(file_name)[0]

            logger.debug(f"Removing annotations for image: {base_name}")

            keys_to_remove = [
                key
                for key in self.mw.all_annotations.keys()
                if key == base_name
                or (
                    key.startswith(f"{base_name}_")
                    and not key.startswith(f"{base_name}_8bit")
                )
            ]

            logger.debug(f"Keys to remove: {keys_to_remove}")

            for key in keys_to_remove:
                del self.mw.all_annotations[key]

            if base_name in self.mw.image_slices:
                # Drop this stack's cached QImages before dropping the list so
                # shared-LRU entries don't leak (issue #45).
                release_slices(self.mw.image_slices[base_name])
                del self.mw.image_slices[base_name]

            # A video never reaches here (redefine only runs for tif/czi), but
            # release its capture defensively so no path can leak one (#47).
            if base_name in self.mw.video_handlers:
                self.mw.video_handlers[base_name].release()
                del self.mw.video_handlers[base_name]

            if self.mw.image_file_name == file_name:
                self.mw.current_image = None
                self.mw.image_label.clear()

            if file_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(file_path, force_dimension_dialog=True)
            elif file_path.lower().endswith(".czi"):
                self.load_czi(file_path, force_dimension_dialog=True)

            self.update_slice_list()
            self.mw.update_annotation_list()
            self.mw.image_label.update()

            QMessageBox.information(
                self.mw,
                "Dimensions Redefined",
                "The dimensions have been redefined and the image reloaded. "
                "All previous annotations for this image have been removed.",
            )

    def remove_image(self):
        current_item = self.mw.image_list.currentItem()
        if current_item:
            file_name = current_item.text()

            self.mw.image_list.takeItem(self.mw.image_list.row(current_item))
            self.mw.image_paths.pop(file_name, None)
            self.mw.all_images = [
                img for img in self.mw.all_images if img["file_name"] != file_name
            ]

            self.mw.all_annotations.pop(file_name, None)

            base_name = os.path.splitext(file_name)[0]
            if base_name in self.mw.image_slices:
                stack_slices = self.mw.image_slices[base_name]
                for slice_name in slice_names(stack_slices):
                    self.mw.all_annotations.pop(slice_name, None)
                release_slices(stack_slices)  # evict cached QImages (issue #45)
                del self.mw.image_slices[base_name]

                # If the removed stack/video was the ACTIVE one, drop the live
                # slice references too -- otherwise update_ui() ->
                # update_slice_list() rebuilds the slice list from the dangling
                # mw.slices and the orphaned frames reappear (and a video's
                # frames can't decode, its handler having just been released).
                if self.mw.slices is stack_slices:
                    self.mw.slices = []

                self.mw.slice_list.clear()

            # Release the video's cv2 capture, if any (issue #47).
            if base_name in self.mw.video_handlers:
                self.mw.video_handlers[base_name].release()
                del self.mw.video_handlers[base_name]

            if self.mw.image_file_name == file_name:
                self.mw.current_image = None
                self.mw.image_file_name = ""
                self.mw.current_slice = None
                self.mw.image_label.clear()
                self.mw.annotation_list.setRowCount(0)

            if self.mw.image_list.count() > 0:
                next_item = self.mw.image_list.item(0)
                self.mw.image_list.setCurrentItem(next_item)
                self.switch_image(next_item)
            else:
                self.mw.current_image = None
                self.mw.image_file_name = ""
                self.mw.current_slice = None
                self.mw.image_label.clear()
                self.mw.annotation_list.setRowCount(0)
                self.mw.slice_list.clear()

            self.mw.update_ui()
            self.mw.auto_save()

    def delete_selected_image(self):
        current_item = self.mw.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            reply = QMessageBox.question(
                self.mw,
                "Delete Image",
                f"Are you sure you want to delete the image '{file_name}'?\n\n"
                "This will remove the image and all its associated annotations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.mw.image_list.takeItem(self.mw.image_list.row(current_item))
                self.mw.image_paths.pop(file_name, None)
                self.mw.all_images = [
                    img for img in self.mw.all_images if img["file_name"] != file_name
                ]

                self.mw.all_annotations.pop(file_name, None)

                base_name = os.path.splitext(file_name)[0]
                if base_name in self.mw.image_slices:
                    stack_slices = self.mw.image_slices[base_name]
                    for slice_name in slice_names(stack_slices):
                        self.mw.all_annotations.pop(slice_name, None)
                    release_slices(stack_slices)  # evict cached QImages (issue #45)
                    del self.mw.image_slices[base_name]

                    # Drop live slice refs if the removed stack/video was the
                    # active one, else update_ui() rebuilds the list from the
                    # dangling mw.slices (orphaned frames reappear). Mirrors
                    # remove_image (video-removal bug).
                    if self.mw.slices is stack_slices:
                        self.mw.slices = []

                    self.mw.slice_list.clear()

                # Release the video's cv2 capture, if any (issue #47).
                if base_name in self.mw.video_handlers:
                    self.mw.video_handlers[base_name].release()
                    del self.mw.video_handlers[base_name]

                if self.mw.image_file_name == file_name:
                    self.mw.current_image = None
                    self.mw.image_file_name = ""
                    self.mw.current_slice = None
                    self.mw.image_label.clear()
                    self.mw.annotation_list.setRowCount(0)

                if self.mw.image_list.count() > 0:
                    next_item = self.mw.image_list.item(0)
                    self.mw.image_list.setCurrentItem(next_item)
                    self.switch_image(next_item)
                else:
                    self.mw.current_image = None
                    self.mw.image_file_name = ""
                    self.mw.current_slice = None
                    self.mw.image_label.clear()
                    self.mw.annotation_list.setRowCount(0)
                    self.mw.slice_list.clear()

                self.mw.update_ui()

                QMessageBox.information(
                    self.mw,
                    "Image Deleted",
                    f"The image '{file_name}' has been deleted.",
                )

    def display_image(self):
        if self.mw.current_image:
            if isinstance(self.mw.current_image, QImage):
                pixmap = QPixmap.fromImage(self.mw.current_image)
            elif isinstance(self.mw.current_image, QPixmap):
                pixmap = self.mw.current_image
            else:
                logger.warning(f"Unexpected image type: {type(self.mw.current_image)}")
                return

            if not pixmap.isNull():
                self.mw.image_label.setPixmap(pixmap)
                self.mw.image_label.adjustSize()
            else:
                logger.warning("Null pixmap")
        else:
            self.mw.image_label.clear()
            logger.debug("No current image to display")
