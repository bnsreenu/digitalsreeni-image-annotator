"""
Unit tests for core.torch_utils device resolution (upstream issue #57).

A fake `torch` module is injected into sys.modules so the tests run the
same on machines with and without CUDA.
"""

import sys
import types

import pytest

from src.digitalsreeni_image_annotator.core import torch_utils


@pytest.fixture(autouse=True)
def reset_cache():
    torch_utils._cached_result = None
    torch_utils._warning_shown = False
    yield
    torch_utils._cached_result = None
    torch_utils._warning_shown = False


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a configurable fake torch module; returns its cuda namespace."""
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_capability=lambda idx=0: (8, 6),
        get_arch_list=lambda: ["sm_70", "sm_80", "sm_90", "compute_90"],
        get_device_name=lambda idx=0: "Fake GPU",
    )
    module = types.ModuleType("torch")
    module.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", module)
    return cuda


def test_no_cuda_returns_cpu_without_warning(fake_torch):
    assert torch_utils.resolve_torch_device() == ("cpu", None)


def test_supported_gpu_returns_cuda(fake_torch):
    fake_torch.is_available = lambda: True
    assert torch_utils.resolve_torch_device() == ("cuda", None)


def test_unsupported_pascal_gpu_falls_back_to_cpu(fake_torch):
    fake_torch.is_available = lambda: True
    fake_torch.get_device_capability = lambda idx=0: (6, 1)  # GTX 1050
    device, warning = torch_utils.resolve_torch_device()
    assert device == "cpu"
    assert "sm_61" in warning
    assert "sm_70" in warning
    assert "Fake GPU" in warning


def test_oldest_supported_capability_is_accepted(fake_torch):
    fake_torch.is_available = lambda: True
    fake_torch.get_device_capability = lambda idx=0: (7, 0)
    assert torch_utils.resolve_torch_device() == ("cuda", None)


def test_empty_arch_list_keeps_cuda(fake_torch):
    # Defensive: if torch reports no compiled arches, don't second-guess it.
    fake_torch.is_available = lambda: True
    fake_torch.get_arch_list = lambda: []
    assert torch_utils.resolve_torch_device() == ("cuda", None)


def test_probe_failure_falls_back_to_cpu(fake_torch):
    fake_torch.is_available = lambda: True

    def boom(idx=0):
        raise RuntimeError("driver mismatch")

    fake_torch.get_device_capability = boom
    device, warning = torch_utils.resolve_torch_device()
    assert device == "cpu"
    assert "driver mismatch" in warning


def test_missing_torch_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → fails
    assert torch_utils.resolve_torch_device() == ("cpu", None)


def test_result_is_cached(fake_torch):
    fake_torch.is_available = lambda: True
    assert torch_utils.resolve_torch_device() == ("cuda", None)
    # Changing the fake afterwards must not change the cached decision.
    fake_torch.is_available = lambda: False
    assert torch_utils.resolve_torch_device() == ("cuda", None)


def test_parse_arch_list():
    assert torch_utils._parse_arch_list(
        ["sm_70", "sm_80", "compute_90", "garbage", "sm_xx"]
    ) == [70, 80]


def test_warning_dialog_shown_once(fake_torch, monkeypatch, qt_application):
    fake_torch.is_available = lambda: True
    fake_torch.get_device_capability = lambda idx=0: (6, 1)

    calls = []
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: calls.append(a)
    )

    torch_utils.maybe_warn_cpu_fallback(None)
    torch_utils.maybe_warn_cpu_fallback(None)
    assert len(calls) == 1


def test_no_warning_dialog_when_device_ok(fake_torch, monkeypatch, qt_application):
    fake_torch.is_available = lambda: True

    calls = []
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: calls.append(a)
    )

    torch_utils.maybe_warn_cpu_fallback(None)
    assert calls == []


class TestIsOom:
    """``_is_oom`` classifies out-of-memory failures torch-free (issue #34)."""

    def test_memoryerror_is_oom(self):
        assert torch_utils._is_oom(MemoryError()) is True

    def test_cuda_out_of_memory_message_is_oom(self):
        assert torch_utils._is_oom(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        ) is True

    def test_not_enough_memory_message_is_oom(self):
        assert torch_utils._is_oom(RuntimeError("not enough memory")) is True

    def test_plain_runtimeerror_is_not_oom(self):
        assert torch_utils._is_oom(RuntimeError("weights corrupt")) is False

    def test_wrong_type_with_oom_text_is_not_oom(self):
        # Only MemoryError or RuntimeError qualify; the message text alone on a
        # different exception type must not be treated as OOM.
        assert torch_utils._is_oom(ValueError("out of memory")) is False
