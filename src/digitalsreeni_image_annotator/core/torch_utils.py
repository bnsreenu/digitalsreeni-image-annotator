"""
Shared torch device selection (upstream issue #57).

torch.cuda.is_available() returns True for any visible CUDA device, even
when the installed torch wheels contain no kernels for its compute
capability (e.g. torch >= 2.8 wheels ship sm_70+ only, so a Pascal
GTX 1050 / sm_61 passes the availability check but every kernel launch
dies with "CUDA error: no kernel image is available for execution on
the device"). This module detects that mismatch up front and falls back
to CPU with an actionable warning instead of a cryptic crash mid-inference.
"""
from .logging_config import get_logger

logger = get_logger(__name__)

_cached_result = None


def resolve_torch_device():
    """Return ``(device, warning)``.

    ``device`` is ``"cuda"`` or ``"cpu"``; ``warning`` is None or a
    human-readable explanation of why CUDA was rejected (printed once on
    first call). The result is cached for the process lifetime so SAM,
    DINO and YOLO all share one decision.
    """
    global _cached_result
    if _cached_result is None:
        _cached_result = _resolve()
        if _cached_result[1]:
            logger.warning(_cached_result[1])
    return _cached_result


def _resolve():
    try:
        import torch
    except Exception:
        return ("cpu", None)

    try:
        if not torch.cuda.is_available():
            return ("cpu", None)

        # Deliberately device-0-only: on a mixed multi-GPU box this may
        # force CPU even though a later index is supported; the app never
        # selects a non-default CUDA device anywhere, so index 0 is what
        # inference would actually run on.
        major, minor = torch.cuda.get_device_capability(0)
        device_sm = major * 10 + minor
        compiled_sms = _parse_arch_list(torch.cuda.get_arch_list())

        if compiled_sms and device_sm < min(compiled_sms):
            gpu = torch.cuda.get_device_name(0)
            return (
                "cpu",
                f"GPU '{gpu}' (compute capability sm_{device_sm}) is not "
                f"supported by the installed PyTorch build (compiled for "
                f"sm_{min(compiled_sms)}+). Falling back to CPU. For GPU "
                f"inference install an older PyTorch with support for this "
                f"card, e.g.:\n"
                f"  pip install torch==2.4.1 torchvision==0.19.1 "
                f"--index-url https://download.pytorch.org/whl/cu121",
            )
        return ("cuda", None)
    except Exception as e:
        # Any probing failure: prefer a working CPU path over a crash.
        return ("cpu", f"Could not verify CUDA compatibility ({e}); "
                       f"falling back to CPU.")


_warning_shown = False


def maybe_warn_cpu_fallback(parent=None):
    """Show the CUDA-incompatibility warning as a dialog, once per session.

    No-op when the device resolved cleanly (CUDA usable, or no GPU at
    all — running on CPU without a discrete GPU is expected and needs
    no dialog).
    """
    global _warning_shown
    _, warning = resolve_torch_device()
    if warning is None or _warning_shown:
        return
    _warning_shown = True
    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.warning(parent, "GPU not usable — running on CPU", warning)


def _parse_arch_list(arch_list):
    """Extract numeric sm values from e.g. ["sm_70", "sm_80", "compute_90"]."""
    sms = []
    for arch in arch_list:
        prefix, _, num = arch.rpartition("_")
        if prefix.endswith("sm") and num.isdigit():
            sms.append(int(num))
    return sms


def _is_oom(exc) -> bool:
    """True for CUDA or host out-of-memory errors, without importing torch.

    ``torch.cuda.OutOfMemoryError`` is a ``RuntimeError`` subclass whose message
    contains "out of memory", so the string check covers it torch-free.
    """
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "cuda oom" in text or "not enough memory" in text
    )
