"""
SAM utilities — delegates to an isolated subprocess to avoid DLL conflicts.

On Windows + Python 3.14, loading PyTorch after PyQt5 causes
WinError 1114. Running SAM in a clean subprocess avoids the issue.
"""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage


MODEL_NAMES = [
    "SAM 2 tiny",
    "SAM 2 small",
    "SAM 2 base",
    "SAM 2 large",
    "SAM 2.1 tiny",
    "SAM 2.1 small",
    "SAM 2.1 base",
    "SAM 2.1 large",
]

MODEL_FILES = {
    "SAM 2 tiny": "sam2_t.pt",
    "SAM 2 small": "sam2_s.pt",
    "SAM 2 base": "sam2_b.pt",
    "SAM 2 large": "sam2_l.pt",
    "SAM 2.1 tiny": "sam2.1_t.pt",
    "SAM 2.1 small": "sam2.1_s.pt",
    "SAM 2.1 base": "sam2.1_b.pt",
    "SAM 2.1 large": "sam2.1_l.pt",
}


def _qimage_to_numpy(qimage):
    """QImage → RGB numpy array."""
    width = qimage.width()
    height = qimage.height()
    fmt = qimage.format()

    if fmt == QImage.Format_Grayscale8:
        buffer = qimage.constBits().asarray(height * width)
        img = np.frombuffer(buffer, np.uint8).reshape((height, width))
        return np.stack((img,) * 3, -1)

    if fmt in (QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied):
        buffer = qimage.constBits().asarray(height * width * 4)
        img = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
        return img[:, :, :3]

    if fmt == QImage.Format_RGB888:
        buffer = qimage.constBits().asarray(height * width * 3)
        return np.frombuffer(buffer, np.uint8).reshape((height, width, 3))

    # Fallback
    converted = qimage.convertToFormat(QImage.Format_RGB32)
    buffer = converted.constBits().asarray(height * width * 4)
    img = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
    return img[:, :, :3]


class SAMUtils:
    """Thin wrapper that forwards SAM work to a subprocess worker."""

    # Exposed for backward compat with annotator_window.py UI setup
    sam_models = MODEL_FILES.copy()

    def __init__(self):
        self.current_sam_model = None
        # Invoke the worker script directly so the package __init__.py
        # (which imports PyQt5) does not run inside the subprocess.
        self._worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sam_worker.py"
        )

    def change_sam_model(self, model_name):
        if model_name == "Pick a SAM Model":
            self.current_sam_model = None
            print("SAM model unset")
            return

        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown SAM model: {model_name}")

        self.current_sam_model = model_name
        print(f"Selected SAM model: {model_name}")

    def _send_request(self, request: dict) -> dict:
        """Spawn the SAM worker, send JSON, and return parsed response."""
        env = os.environ.copy()
        # Propagate the virtual environment
        for possible in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            v = os.environ.get(possible)
            if v:
                env[possible] = v
                break

        proc = subprocess.run(
            [sys.executable, self._worker_script],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            env=env,
        )

        if proc.returncode != 0:
            err_text = proc.stderr.strip() if proc.stderr else "(no stderr)"
            raise RuntimeError(
                f"SAM worker exited with code {proc.returncode}.\nstderr: {err_text}"
            )

        # Echo worker stdout (includes GPU/CPU diagnostics) to parent console
        lines = (proc.stdout or "").strip().splitlines()
        for line in lines[:-1]:
            print(line)

        try:
            return json.loads(lines[-1])
        except (json.JSONDecodeError, IndexError):
            out_text = proc.stdout.strip() if proc.stdout else "(no stdout)"
            raise RuntimeError(
                f"SAM worker returned non-JSON output.\nstdout: {out_text}"
            )

    @staticmethod
    def _save_image_temp(image: QImage) -> str:
        """Convert QImage to a temporary file and return the path."""
        arr = _qimage_to_numpy(image)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr).save(tmp.name)
        tmp.close()
        return tmp.name

    def apply_sam_points(self, image, positive_points, negative_points):
        if not self.current_sam_model:
            print("No SAM model selected.")
            return None
        try:
            tmp_path = self._save_image_temp(image)
            request = {
                "image_path": tmp_path,
                "model_name": self.current_sam_model,
                "points": {
                    "positive": [list(p) for p in positive_points],
                    "negative": [list(p) for p in negative_points],
                },
            }
            result = self._send_request(request)
        except Exception:
            traceback.print_exc()
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if "error" in result:
            print(f"SAM worker error: {result['error']}")
            return None

        return {
            "segmentation": result["segmentation"],
            "score": result["score"],
        }

    def apply_sam_prediction(self, image, bbox):
        if not self.current_sam_model:
            print("No SAM model selected.")
            return None
        try:
            tmp_path = self._save_image_temp(image)
            request = {
                "image_path": tmp_path,
                "model_name": self.current_sam_model,
                "bboxes": list(bbox),
            }
            result = self._send_request(request)
        except Exception:
            traceback.print_exc()
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if "error" in result:
            print(f"SAM worker error: {result['error']}")
            return None

        return {
            "segmentation": result["segmentation"],
            "score": result["score"],
        }
