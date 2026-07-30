"""Trained-model sidecar: what a ``.pt`` cannot tell you about itself (#74).

A checkpoint carries weights and a task, but not the class names it was trained
on, not the keypoint schema behind its ``kpt_shape``, not the configuration it
was produced with, and not how well it did. All of that lived only in the
session that produced it, which is why using a freshly trained model meant
re-loading it by hand and then guessing.

The sidecar is a JSON file next to the weights. It is **additive**: a ``.pt``
without one still loads through the existing bare-``kpt_shape`` reconstruction
path, so models trained outside the app keep working (ADR-029 PR-3). Reading
one is a strict improvement, never a requirement.

Qt-free so the CLI (issue #76) can read a sidecar without a display.
"""

import json
import os

SIDECAR_SUFFIX = ".json"


def sidecar_path(weights_path):
    """Path of the sidecar belonging to ``weights_path``."""
    return os.path.splitext(str(weights_path))[0] + SIDECAR_SUFFIX


def build_sidecar(
    *,
    model_type,
    task=None,
    class_names=None,
    keypoint_schema=None,
    kpt_shape=None,
    flip_idx=None,
    config=None,
    metrics=None,
    timestamp=None,
):
    """Assemble the sidecar payload.

    Keys whose value is absent are **omitted rather than written as null**: a
    reader distinguishing "not applicable" from "recorded as nothing" has an
    easier job, and the file stays readable by a human.
    """
    payload = {"schema_version": 1, "model_type": model_type}
    optional = {
        "task": task,
        "class_names": list(class_names) if class_names else None,
        "keypoint_schema": keypoint_schema,
        "kpt_shape": list(kpt_shape) if kpt_shape else None,
        "flip_idx": list(flip_idx) if flip_idx else None,
        "config": config or None,
        "metrics": metrics or None,
        "trained_at": timestamp,
    }
    payload.update({key: value for key, value in optional.items() if value})
    return payload


def write_sidecar(weights_path, payload):
    """Write ``payload`` beside ``weights_path``. Returns the path written.

    Raises on failure rather than swallowing: a sidecar that silently failed to
    write would make the model look externally-trained later, and the
    difference is invisible until someone wonders where their class names went.
    Callers at the UI boundary catch and report (ADR-031).
    """
    path = sidecar_path(weights_path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def read_sidecar(weights_path):
    """Sidecar payload for ``weights_path``, or ``None``.

    ``None`` covers both "no sidecar" and "unreadable sidecar", and the caller
    treats them identically: fall back to the bare-``kpt_shape`` reconstruction
    that externally-trained models already use. A corrupt sidecar must not be
    worse than no sidecar.
    """
    path = sidecar_path(weights_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def unique_weights_path(directory, base_name, timestamp):
    """A non-colliding ``<directory>/<base>_<timestamp>.pt``.

    The timestamp makes collisions unlikely, not impossible — two runs finishing
    inside the same second are entirely possible on a fast machine with a small
    dataset. Handled rather than assumed away, because the failure mode is
    overwriting a model the user may not have copied anywhere yet.
    """
    os.makedirs(directory, exist_ok=True)
    stem = f"{base_name}_{timestamp}"
    candidate = os.path.join(directory, stem + ".pt")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{counter}.pt")
        counter += 1
    return candidate


def format_metrics(metrics):
    """``[(label, formatted_value), ...]`` for the results panel.

    A metric that is unavailable for a given training path is **omitted, not
    shown as a placeholder** — an empty row reads as "the model scored nothing",
    which is a very different claim from "this path does not report that".
    """
    labels = {
        "mAP50": "mAP@50",
        "mAP50-95": "mAP@50-95",
        "precision": "Precision",
        "recall": "Recall",
        "best_epoch": "Best epoch",
        "final_loss": "Final loss",
        "mean_iou": "Mean IoU",
    }
    rows = []
    for key, label in labels.items():
        value = (metrics or {}).get(key)
        if value is None:
            continue
        if isinstance(value, float):
            rows.append((label, f"{value:.4f}"))
        else:
            rows.append((label, str(value)))
    return rows
