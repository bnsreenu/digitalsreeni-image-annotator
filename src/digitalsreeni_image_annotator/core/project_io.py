"""Read an ``.iap`` project without a GUI (issue #76, ADR-041).

``ProjectController.load_project_data`` does far more than read a file: it
populates widgets, restores DINO panels, switches to the first image, and sits
behind the autosave and recovery machinery. None of that is meaningful
headlessly, and all of it needs Qt.

This module extracts the part a script actually wants — annotations, classes,
image paths — as plain data.

**Read-only by construction.** There is no write path here at all. The CLI must
never produce an autosave or a recovery snapshot in a project it was pointed
at: those exist to protect interactive editing (ADR-005/032), and a build
script silently rewriting the file it was asked to read is exactly the surprise
a CI gate must not spring.

Path resolution mirrors :meth:`ProjectController.resolve_image_path` —
relative-first, then absolute, then the historical convention (ADR-033). That
order is what makes a project portable to a CI runner, which is the whole point
of having the CLI resolve paths at all.
"""

import json
import os

from .logging_config import get_logger

logger = get_logger(__name__)


class ProjectReadError(RuntimeError):
    """The project file could not be read or is not an ``.iap`` project."""


class LoadedProject:
    """The Qt-free view of a project.

    Attribute names match the main window's (``all_annotations``,
    ``class_mapping``, ``image_paths``, ``all_images``) so the export functions,
    which were written against those, take this object's fields unchanged.
    """

    def __init__(self, path, data):
        self.path = path
        self.directory = os.path.dirname(os.path.abspath(path))
        self.data = data

        self.all_images = list(data.get("images") or [])
        self.class_mapping = {}
        self.keypoint_schemas = {}
        for index, class_info in enumerate(data.get("classes") or [], start=1):
            name = class_info.get("name")
            if not name:
                continue
            self.class_mapping[name] = class_info.get("id", index)
            schema = class_info.get("keypoint_schema")
            if schema:
                self.keypoint_schemas[name] = schema

        self.all_annotations = {}
        for image_info in self.all_images:
            if image_info.get("is_multi_slice"):
                for slice_info in image_info.get("slices") or []:
                    self.all_annotations[slice_info["name"]] = (
                        slice_info.get("annotations") or {}
                    )
            else:
                self.all_annotations[image_info["file_name"]] = (
                    image_info.get("annotations") or {}
                )

        self.image_paths = {}
        self.missing_images = []
        for image_info in self.all_images:
            file_name = image_info.get("file_name")
            if not file_name:
                continue
            resolved = self.resolve_image_path(file_name)
            if resolved is None:
                self.missing_images.append(file_name)
            else:
                self.image_paths[file_name] = resolved

    def resolve_image_path(self, file_name):
        """Relative-first path resolution (ADR-033), or ``None``."""
        rel = (self.data.get("image_paths_rel") or {}).get(file_name)
        if rel:
            candidate = os.path.normpath(os.path.join(self.directory, rel))
            if os.path.exists(candidate):
                return candidate

        absolute = (self.data.get("image_paths") or {}).get(file_name)
        if absolute and os.path.exists(absolute):
            return absolute

        candidate = os.path.join(self.directory, "images", file_name)
        if os.path.exists(candidate):
            return candidate
        return None

    def image_sizes(self):
        """``{image_or_slice_name: (width, height)}`` from the recorded shapes.

        Read from the project rather than from disk: the QC bounds rules need
        sizes, and opening every image to get them would make ``validate``
        slow for no benefit on a project that already recorded them.
        """
        sizes = {}
        for image_info in self.all_images:
            file_name = image_info.get("file_name")
            width = image_info.get("width")
            height = image_info.get("height")
            if file_name and width and height:
                sizes[file_name] = (int(width), int(height))
            shape = image_info.get("shape")
            if file_name and shape and len(shape) >= 2:
                size = (int(shape[-1]), int(shape[-2]))
                sizes.setdefault(file_name, size)
                for slice_info in image_info.get("slices") or []:
                    sizes.setdefault(slice_info["name"], size)
        return sizes

    def class_names(self):
        return list(self.class_mapping)

    def slice_names(self):
        """Names of every materialised slice recorded in the project.

        The CLI supports what the project **already materialised**; extracting
        new slices from a stack runs through ``ImageController``, which is
        Qt-bound. That limit is documented rather than half-supported.
        """
        names = []
        for image_info in self.all_images:
            for slice_info in image_info.get("slices") or []:
                names.append(slice_info["name"])
        return names


def load_project(path):
    """Read an ``.iap`` file into a :class:`LoadedProject`. Read-only."""
    if not os.path.exists(path):
        raise ProjectReadError(f"No such project file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ProjectReadError(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict) or "images" not in data:
        raise ProjectReadError(
            f"{path} does not look like an .iap project (no 'images' key)."
        )
    return LoadedProject(path, data)
