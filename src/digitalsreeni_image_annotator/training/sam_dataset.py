"""Build SAM fine-tuning :class:`SampleGroup`s from either the live project
annotations or a prepared on-disk dataset folder.

The project path mirrors the image-resolution logic in
``io.export_formats.export_yolo_v5plus`` (slice lookup via ``slices`` /
``image_slices``; regular images via ``image_paths`` with exact-then-substring
match; TIFF/CZI source files skipped in favour of their extracted slices), so a
dataset that exports cleanly to YOLO also trains cleanly here.
"""

from __future__ import annotations

import json
import os

from PyQt6.QtGui import QImage

from .sam_trainer import SampleGroup
from ..core.slice_index import resolve_slice_image as _resolve_slice_image
from ..core.slice_index import slice_index as _slice_index
from ..inference.sam_utils import _qimage_to_numpy

from ..core.logging_config import get_logger

logger = get_logger(__name__)


def _specs_for(annotations) -> list:
    """Flatten ``{class: [ann, ...]}`` into raw instance specs the
    :class:`SampleGroup` rasterises lazily."""
    specs = []
    for _class_name, class_annotations in (annotations or {}).items():
        for ann in class_annotations:
            if ann.get("segmentation"):
                specs.append({"segmentation": ann["segmentation"]})
            elif ann.get("bbox"):
                specs.append({"bbox": ann["bbox"]})
    return specs


def build_groups_from_project(all_annotations, image_paths, slices, image_slices):
    """Live project annotations → ``list[SampleGroup]``.

    Images load lazily (one at a time during training) to bound memory; in-RAM
    slice QImages are reused directly.
    """
    slice_index = _slice_index(slices, image_slices)
    groups = []

    for image_name, image_annotations in all_annotations.items():
        specs = _specs_for(image_annotations)
        if not specs:
            continue

        if image_name in slice_index or ("_" in image_name and "." not in image_name):
            qimage = _resolve_slice_image(slice_index, image_name)
            if qimage is None:
                logger.warning(f"skip slice {image_name!r}: no image data")
                continue
            # Convert the in-memory slice QImage to numpy HERE, on the GUI
            # thread. The array is later consumed by the training worker
            # thread; reading constBits() of a live, GUI-shared QImage from
            # another thread is exactly what _qimage_to_numpy warns against,
            # so we hand the worker a thread-owned copy instead of a lambda
            # that defers the buffer read onto the worker.
            arr = _qimage_to_numpy(qimage)
            groups.append(SampleGroup(lambda a=arr: a, specs, name=image_name))
            continue

        image_path = image_paths.get(image_name)
        if image_path is None:
            image_path = next(
                (p for name, p in image_paths.items() if image_name in name), None
            )
        if not image_path:
            logger.warning(f"skip {image_name!r}: no image_paths entry")
            continue
        if image_path.lower().endswith((".tif", ".tiff", ".czi")):
            logger.debug(f"skip TIFF/CZI source {image_name!r} (use slices)")
            continue
        groups.append(SampleGroup(lambda p=image_path: _qimage_to_numpy(QImage(p)), specs, name=image_name))

    return groups


# ── prepared folder ──────────────────────────────────────────────────────────

def build_groups_from_folder(folder: str):
    """Read a folder produced by ``export_sam_dataset`` → ``list[SampleGroup]``.

    Expects ``<folder>/manifest.json`` with entries
    ``{"image": "images/x.png", "instances": [{"bbox": [...]}|{"segmentation": [...]}]}``.
    """
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {folder}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    groups = []
    for entry in manifest.get("images", []):
        img_rel = entry["image"]
        img_path = os.path.join(folder, img_rel)
        specs = entry.get("instances", [])
        if not specs or not os.path.exists(img_path):
            continue
        # Ext-stripped basename, matching what the project path puts in `name`
        # (`build_groups_from_project` uses the annotation key). The manifest
        # stores `images/clip_F00042.png`, and the dot in that made
        # `derive_groups` treat every frame of a recording as its own group --
        # so "Fine-Tune SAM from Dataset Folder" silently got no grouping at
        # all while the project path was correctly grouped. Normalising here
        # rather than at the split keeps every consumer of `name` seeing one
        # shape.
        groups.append(SampleGroup(
            lambda p=img_path: _qimage_to_numpy(QImage(p)),
            specs,
            name=os.path.splitext(os.path.basename(img_rel))[0],
        ))
    return groups


# ── train/val split ──────────────────────────────────────────────────────────

def split_groups(groups, train_pct, keyed_groups=None):
    """Partition ``groups`` into ``(train, val)`` deterministically by source.

    ``train_pct`` in ``[0, 100]``; ``>= 100`` (or fewer than 2 groups) keeps
    everything in train with an empty val set — the caller then skips the
    validation pass / early stopping. Reuses ``io.export_formats.assign_train_val``
    (stable MD5 ordering) so the SAM split matches the YOLO export's behaviour
    and is reproducible across runs and machines.

    Each group is keyed by ``"{index}:{name}"``, which keeps two same-named
    ``SampleGroup``s distinct as *entries*. They do now share a split bucket,
    deliberately: identically-named sources are exactly what the grouping is
    supposed to keep together. An **empty** name falls back to the unique key,
    since "unnamed" is not evidence of a shared source.

    **Split by source, not by name (ADR-044).** ``SampleGroup.name`` is the
    source image or slice name, so a stack's slices and a video's frames are
    routed to one side together instead of straddling the split — otherwise the
    val loss is measured on frames all but identical to trained ones, and early
    stopping is driven by a number that means nothing.

    The grouping comes from the names alone: this runs on the training worker
    thread and has no access to the main window's ``image_slices``, so
    ``keyed_groups`` lets the GUI hand over the grouping it already computed
    and *warned about*, refined by near-duplicate clusters when a curation run
    produced any (ADR-045). Re-deriving it here instead would silently drop
    that refinement, so the dialog would describe one split and the run would
    perform another. Keys it does not recognise are ignored and keys it omits
    fall back to the derived grouping, so a stale mapping degrades to the
    structural split rather than to nonsense.

    Otherwise the grouping comes from the names alone: this runs on the
    training worker thread with no access to the main window's
    ``image_slices``, so ``derive_groups`` falls back to its name-prefix rule — which covers every
    name the app itself produces.
    """
    from ..io.export_formats import assign_train_val

    groups = list(groups)
    if train_pct >= 100 or len(groups) < 2:
        return groups, []

    keyed, derived = split_keys(groups)
    if keyed_groups:
        if not set(keyed_groups) & set(derived):
            # The keys are "{index}:{name}", rebuilt here from the same list the
            # caller keyed. If they ever stop matching -- a reordered or
            # rebuilt group list -- every lookup falls back and the refinement
            # vanishes without a trace. Say so; the fallback is safe, the
            # silence is not.
            logger.warning(
                "the supplied grouping matched none of the %d split keys; "
                "using the derived grouping instead",
                len(derived),
            )
        derived = {
            key: keyed_groups.get(key, group) for key, group in derived.items()
        }
    _train_keys, val_keys = assign_train_val(
        keyed.keys(), 100 - train_pct, derived
    )
    train = [g for k, g in keyed.items() if k not in val_keys]
    val = [g for k, g in keyed.items() if k in val_keys]
    return train, val


def split_keys(groups):
    """``({split key: group}, {split key: source group})`` for ``groups``.

    Factored out so the warning shown before a run previews **this** mapping
    rather than rebuilding an approximation of it. Passing the bare
    ``[g.name for g in groups]`` looked equivalent and was not: two groups can
    share a name (a prepared folder holding `a.png` and `a.jpg` ext-strips both
    to `a`), and a list of duplicates collapses to one entry before the split
    sees it — so the preview reported a healthy split while the real one
    degenerated and fell back. Same divergence class as the export preview,
    one layer down.
    """
    from ..core.dataset_split import derive_groups

    keyed = {f"{index}:{group.name}": group for index, group in enumerate(groups)}
    name_groups = derive_groups([group.name for group in groups])

    keyed_groups = {}
    for key, group in keyed.items():
        # An unnamed group falls back to its own unique key: collapsing every
        # `name=""` group into one bucket is exactly what the indexed key
        # exists to prevent.
        keyed_groups[key] = (name_groups.get(group.name) or key) if group.name else key
    return keyed, keyed_groups
