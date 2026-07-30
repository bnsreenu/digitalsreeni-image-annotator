"""
Phase 0 gate: confirm PyQt6 and PyTorch can coexist in one process.

Why this exists
---------------
The historical ADR-011 documented that on Windows + Python 3.14,
importing PyQt first and then loading PyTorch triggers
``WinError 1114`` (DLL load-order conflict). It was thought that
migrating to PyQt6 (ADR-014) eliminated the conflict, but
real-world testing with torch 2.11.0 + PyQt6 6.10.2 shows the
conflict still surfaces when Qt DLLs are loaded BEFORE torch.
The workaround is simple and confirmed: import torch eagerly
before QApplication is created so torch claims its DLL slot first.
See ADR-017.

The crucial bit: plain ``import PyQt6`` does NOT load Qt's native
platform plugin (qwindows.dll on Windows, libqxcb on Linux).  The
plugin is loaded lazily by ``QApplication.__init__``.  So this
script tests BOTH orders to document the real failure mode and
confirm safe order.

Usage
-----
    python tools/check_pyqt6_torch_coexistence.py

Exit code 0 means torch-first works (production order).
Exit code 1 means torch-first also fails → return to subprocess.
"""

from __future__ import annotations

import platform
import sys
import traceback


def _try(label: str, fn) -> bool:
    print(f"[{label}] running ...", flush=True)
    try:
        result = fn()
    except BaseException:  # catch SystemExit / segfault recovery too
        print(f"[{label}] FAILED:")
        traceback.print_exc()
        return False
    if result is not None and hasattr(result, "__version__"):
        print(f"[{label}] OK — version: {result.__version__}", flush=True)
    else:
        print(f"[{label}] OK", flush=True)
    return True


def _construct_qapplication():
    """Force Qt's platform plugin to load.

    On Windows this is where qwindows.dll gets loaded, which is the
    site of the historical WinError 1114. We use 'offscreen' so the
    script runs in a headless CI / SSH context.
    """
    import os
    # Don't clobber an existing user setting — they may want to test
    # the real platform plugin specifically.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def _check_torch_then_qt() -> bool:
    """
    Production import order: torch first, then QApplication.
    This is what main.py does (see ADR-017).
    """
    ok = True
    ok &= _try("(torch-first) torch", lambda: __import__("torch"))
    ok &= _try("(torch-first) torchvision", lambda: __import__("torchvision"))
    ok &= _try("(torch-first) transformers", lambda: __import__("transformers"))
    ok &= _try("(torch-first) ultralytics", lambda: __import__("ultralytics"))
    ok &= _try(
        "(torch-first) QApplication construct (loads Qt platform plugin)",
        _construct_qapplication,
    )
    return ok


def _check_qt_then_torch() -> bool:
    """
    The import order that ADR-014 thought was fixed.  On some torch
    versions this still fails (WinError 1114).  We check it so we
    can warn if the 'safe' environment regressed.
    """
    ok = True
    ok &= _try("(qt-first) PyQt6.QtCore", lambda: __import__("PyQt6.QtCore", fromlist=["QtCore"]))
    ok &= _try("(qt-first) PyQt6.QtWidgets", lambda: __import__("PyQt6.QtWidgets", fromlist=["QtWidgets"]))
    ok &= _try("(qt-first) PyQt6.QtGui", lambda: __import__("PyQt6.QtGui", fromlist=["QtGui"]))
    # Force platform plugin load BEFORE torch — this is where the
    # failure appeared.
    ok &= _try(
        "(qt-first) QApplication construct (loads Qt platform plugin)",
        _construct_qapplication,
    )
    if not ok:
        print("[qt-first] Qt failed to load — can't test torch-after-Qt.")
        return False
    ok &= _try("(qt-first) torch", lambda: __import__("torch"))
    ok &= _try("(qt-first) torchvision", lambda: __import__("torchvision"))
    ok &= _try("(qt-first) ultralytics", lambda: __import__("ultralytics"))
    return ok


def main() -> int:
    print(f"Python:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print("-" * 60)

    safe_ok = _check_torch_then_qt()
    print("-" * 60)

    # We run qt-first check in a FRESH process because the preceding
    # torch-first test may have already loaded DLLs that would mask
    # the issue.
    print("\nChecking Qt-first order in a fresh subprocess...")
    import subprocess as sp

    worker_src = """
import ast, sys, traceback


def _try(label, fn):
    print(f"[{label}] running ...", flush=True)
    try:
        result = fn()
    except BaseException:
        print(f"[{label}] FAILED:")
        traceback.print_exc()
        return False
    print(f"[{label}] OK", flush=True)
    return True


def _construct_qapplication():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def _check_qt_then_torch():
    ok = True
    ok &= _try("(qt-first) PyQt6.QtCore", lambda: __import__("PyQt6.QtCore", fromlist=["QtCore"]))
    ok &= _try("(qt-first) PyQt6.QtWidgets", lambda: __import__("PyQt6.QtWidgets", fromlist=["QtWidgets"]))
    ok &= _try("(qt-first) PyQt6.QtGui", lambda: __import__("PyQt6.QtGui", fromlist=["QtGui"]))
    ok &= _try("(qt-first) QApplication", _construct_qapplication)
    if not ok:
        print("[qt-first] Qt failed — can't test torch-after-Qt.")
        return False
    ok &= _try("(qt-first) torch", lambda: __import__("torch"))
    ok &= _try("(qt-first) torchvision", lambda: __import__("torchvision"))
    ok &= _try("(qt-first) ultralytics", lambda: __import__("ultralytics"))
    return ok

ok = _check_qt_then_torch()
print("OK" if ok else "FAIL")
"""
    proc = sp.run(
        [sys.executable, "-c", worker_src],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    qt_first_ok = proc.stdout.strip().endswith("OK") and proc.returncode == 0

    print("-" * 60)
    if not safe_ok:
        print("RESULT: torch-first order FAILED.")
        print("        Return to subprocess isolation (ADR-011).")
        return 1
    if not qt_first_ok:
        print("RESULT: torch-first OK.  Qt-first FAILED (known with some versions).")
        print("        Keep main.py eager torch import (ADR-017).")
    else:
        print("RESULT: both orders clean.  Qt packaging has fixed the conflict.")
        print("        Consider removing main.py eager torch import if confirmed stable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
