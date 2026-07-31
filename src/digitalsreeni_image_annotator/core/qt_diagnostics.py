"""Explain a failed PyQt6 / Qt import in terms the user can act on (issue #92, ADR-046).

Why this exists
---------------
Upgrading PyQt6 inside an *existing Conda* environment on Windows produces::

    ImportError: DLL load failed while importing QtCore:
    The specified procedure could not be found.

with Windows exception ``0xc0000139`` (``STATUS_ENTRYPOINT_NOT_FOUND``). That status
means a DLL imported a function name that the DLL the loader actually resolved does
not export -- i.e. **the wrong copy of a dependency won the search**. It is not a
defect in any particular PyQt6 release, which is why "downgrade PyQt6" appears to fix
it: the binding stops being newer than the Qt runtime it collided with.

How Qt is actually located
--------------------------
The search order below is not a general account of the Windows loader. It emulates
``PyQt6/__init__.py::find_qt()``, which runs at ``import PyQt6`` and decides the
question before the loader is ever consulted::

    dll_dir = os.path.dirname(sys.executable)
    if not os.path.isfile(dll_dir + '\\Qt6Core.dll'):
        dll_dir = os.path.dirname(__file__) + '\\Qt6\\bin'
        if os.path.isfile(dll_dir + '\\Qt6Core.dll'):
            os.environ['PATH'] = dll_dir + ';' + os.environ['PATH']
        else:
            for dll_dir in os.environ['PATH'].split(';'):
                if os.path.isfile(dll_dir + '\\Qt6Core.dll'):
                    break
            else:
                return
    os.add_dll_directory(dll_dir)

Three consequences drive everything here:

* The **interpreter's own directory wins outright**, and when it does, the Qt the
  wheel ships is never registered at all. For a Conda environment that directory *is*
  the environment root.
* ``PATH`` **is** consulted -- but only when the wheel ships no Qt of its own. That is
  how ``%CONDA_PREFIX%\\Library\\bin`` (installed by ``qt6-main``, pulled in by
  ``pyqt``, ``qtpy``, ``spyder``, ``napari``, matplotlib's Qt backend, ...) becomes the
  registered directory. CPython >= 3.8 ignores ``PATH`` when resolving an extension
  module's dependencies, so it would be easy to conclude it cannot matter; PyQt6 puts
  it back.
* The directory holding ``QtCore.pyd`` is searched ahead of all of that, via
  ``LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR``. Dropping a DLL into ``site-packages/PyQt6/``
  is a common response to this very error, so it is checked first.

A second, independent mechanism produces the same status code from a different file:
Conda's ``vc14_runtime`` / ``vs2015_runtime`` ship ``msvcp140.dll`` into the
environment, and newer Qt builds import STL symbols an older copy does not export.
That one cannot be confirmed without loading the DLL, so it is reported as a
``suspect`` rather than asserted.

Constraints
-----------
**This module must never import PyQt6, at module level or inside a function.** It runs
*because* importing Qt failed, and it is reached from ``cli/``, which is bound by the
Qt-free rule (ADR-041) and covered by the subprocess test in
``tests/integration/test_cli.py``.

It also never *loads* a DLL. :func:`dll_file_version` reads the PE version resource
out of the file, so it is safe to call on the very DLL that is crashing the process.

All output is ASCII only, matching the convention documented in ``cli/main.py``: a
Windows console under a legacy code page turns non-ASCII into mojibake exactly when
the output is redirected, which is what pasting a bug report does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

QT_CORE_DLL = "Qt6Core.dll"
# vcruntime140_1.dll hosts __CxxFrameHandler4 and is the classic STATUS_ENTRYPOINT_-
# NOT_FOUND culprit in a mixed-toolchain environment, so it belongs here even though
# msvcp140.dll is the one people have heard of.
MSVC_RUNTIME_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll",
                     "vcruntime140_1.dll")

BINDING_DIST = "PyQt6"
RUNTIME_DIST = "PyQt6-Qt6"
SIP_DIST = "PyQt6-sip"

# Where a Qt6Core.dll was found, labelled for the report.
SOURCE_PACKAGE_DIR = "pyqt6-package-dir"       # holds QtCore.pyd; DLL_LOAD_DIR
SOURCE_PYTHON_DIR = "python-dir"               # dirname(sys.executable)
SOURCE_WHEEL = "pyqt6-wheel"                   # PyQt6/Qt6/bin
SOURCE_CONDA_LIBRARY_BIN = "conda-library-bin"  # a PATH entry, named for clarity
SOURCE_PATH = "path-entry"
SOURCE_SYSTEM32 = "system32"

# Severities. "suspect" exists because the two states are genuinely different and
# collapsing them is how a diagnostic ends up asserting something it cannot know:
#   error   - a mismatch that is proven from two version numbers we read
#   suspect - could well be the cause, but it cannot be confirmed without loading
#             the DLL, which is the one thing this module must not do
#   warning - consistent today, fragile tomorrow
SEVERITY_ERROR = "error"
SEVERITY_SUSPECT = "suspect"
SEVERITY_WARNING = "warning"

# What makes `sreeni-cli doctor` exit non-zero: anything that could explain a Qt that
# will not load. A pure forecast does not.
FAILING_SEVERITIES = (SEVERITY_ERROR, SEVERITY_SUSPECT)


@dataclass(frozen=True)
class Finding:
    """One diagnosed problem. See the severity constants above."""

    severity: str
    title: str
    detail: str
    remedy: str


# --- primitives ------------------------------------------------------------


def _distribution_version(name: str) -> str | None:
    """Installed version of ``name``, read from package *metadata*.

    Deliberately not ``import PyQt6; PyQt6.QtCore.PYQT_VERSION_STR`` -- the import is
    the thing that just failed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - corrupt metadata on a user's machine
        return None


def pyqt6_package_dir() -> str | None:
    """Directory holding the PyQt6 package (and therefore ``QtCore.pyd``), or ``None``.

    ``find_spec`` locates the package without executing it, so this stays safe in an
    environment where importing PyQt6 crashes the interpreter.
    """
    from importlib.util import find_spec

    try:
        spec = find_spec("PyQt6")
    except Exception:  # pragma: no cover - broken meta path finder
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def wheel_qt_bin_dir(package_dir: str | None = None) -> str | None:
    """Directory holding the Qt DLLs the PyQt6 wheel ships, or ``None``.

    ``None`` is normal, not broken: a PyQt6 built against a system Qt (Linux distro
    packages, Homebrew, ``conda install pyqt``) ships no Qt of its own.
    """
    base = package_dir if package_dir is not None else pyqt6_package_dir()
    if not base:
        return None
    candidate = os.path.join(base, "Qt6", "bin")
    return candidate if os.path.isdir(candidate) else None


def dll_file_version(path: str) -> str | None:
    """Four-part file version of a DLL, read from its PE version resource.

    Reads the file; never loads it as a module. Returns ``None`` off Windows, when the
    file carries no version resource, or on any failure -- a diagnostic that raises
    while diagnosing a crash is worse than one that says "unknown".
    """
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class VSFixedFileInfo(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD),
            ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD),
            ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD),
            ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD),
            ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD),
            ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD),
            ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    try:
        # Absolute path, not the bare name. This module's entire premise is that the
        # application directory can shadow a system DLL, and version.dll would resolve
        # through that same order.
        system_dir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        version_dll = ctypes.WinDLL(os.path.join(system_dir, "version.dll"))

        get_size = version_dll.GetFileVersionInfoSizeW
        get_size.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        get_size.restype = wintypes.DWORD
        size = get_size(path, None)
        if not size:
            return None

        get_info = version_dll.GetFileVersionInfoW
        get_info.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p
        ]
        get_info.restype = wintypes.BOOL
        buffer = ctypes.create_string_buffer(size)
        if not get_info(path, 0, size, buffer):
            return None

        query = version_dll.VerQueryValueW
        query.argtypes = [
            ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ]
        query.restype = wintypes.BOOL
        block = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not query(buffer, "\\", ctypes.byref(block), ctypes.byref(length)):
            return None

        info = ctypes.cast(block, ctypes.POINTER(VSFixedFileInfo)).contents
        if info.dwSignature != 0xFEEF04BD:
            return None
        high, low = info.dwFileVersionMS, info.dwFileVersionLS
        return f"{high >> 16}.{high & 0xFFFF}.{low >> 16}.{low & 0xFFFF}"
    except Exception:  # pragma: no cover - any Win32 failure means "unknown"
        return None


def _major_minor(version: str | None) -> tuple[int, int] | None:
    """``"6.11.1.0"`` -> ``(6, 11)``. ``None`` when it cannot be parsed."""
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _version_tuple(version: str | None) -> tuple[int, ...] | None:
    """Full dotted version as a comparable tuple, or ``None`` if unparsable."""
    if not version:
        return None
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


# --- environment capture ---------------------------------------------------


def _conda_prefix() -> str | None:
    """The active Conda prefix, or ``None`` outside Conda.

    ``CONDA_PREFIX`` is only set by an activated shell, so a ``conda-meta`` directory
    next to the interpreter is checked too -- the app is often launched from a
    shortcut or an IDE that never ran ``conda activate``.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        return sys.prefix
    return None


def _system32() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")


def _candidate(directory: str, source: str) -> dict[str, Any] | None:
    """A ``Qt6Core.dll`` in ``directory``, with its version, or ``None``."""
    path = os.path.join(directory, QT_CORE_DLL)
    if not os.path.isfile(path):
        return None
    return {"path": path, "source": source, "version": dll_file_version(path)}


def _path_source(directory: str, conda_prefix: str | None) -> str:
    """Label a ``PATH`` entry, naming Conda's own directory when that is what it is."""
    if conda_prefix:
        library_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.normcase(os.path.abspath(directory)) == os.path.normcase(
            os.path.abspath(library_bin)
        ):
            return SOURCE_CONDA_LIBRARY_BIN
    return SOURCE_PATH


def _qt_core_candidates(conda_prefix: str | None, package_dir: str | None,
                        wheel_bin: str | None) -> list[dict[str, Any]]:
    """Every ``Qt6Core.dll`` that will be consulted, in the order it will be.

    This emulates ``PyQt6/__init__.py::find_qt()`` (quoted in the module docstring)
    rather than describing the Windows loader in general, because ``find_qt`` decides
    the outcome first: it registers exactly **one** directory, and the branch it takes
    determines whether the wheel's own Qt is on the search path at all.
    """
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate: dict[str, Any] | None) -> bool:
        if candidate is None:
            return False
        key = os.path.normcase(os.path.abspath(candidate["path"]))
        if key in seen:
            return False
        seen.add(key)
        found.append(candidate)
        return True

    # 0. The directory holding QtCore.pyd. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR puts it
    #    ahead of everything find_qt registers, and "copy the DLL next to the module"
    #    is a common (wrong) fix people try for this very error.
    if package_dir:
        add(_candidate(package_dir, SOURCE_PACKAGE_DIR))

    # 1. find_qt checks the interpreter's directory FIRST and, if Qt is there, returns
    #    without ever registering the wheel's. For Conda that directory is the env root.
    python_dir = os.path.dirname(sys.executable)
    if add(_candidate(python_dir, SOURCE_PYTHON_DIR)):
        add(_candidate(_system32(), SOURCE_SYSTEM32))
        return found

    # 2. Otherwise the wheel's own Qt, if it ships one.
    if wheel_bin and add(_candidate(wheel_bin, SOURCE_WHEEL)):
        add(_candidate(_system32(), SOURCE_SYSTEM32))
        return found

    # 3. No Qt in the wheel: find_qt walks PATH and registers the first entry that has
    #    one. This is the branch that reaches %CONDA_PREFIX%\Library\bin.
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry:
            continue
        if add(_candidate(entry, _path_source(entry, conda_prefix))):
            break

    add(_candidate(_system32(), SOURCE_SYSTEM32))
    return found


def _msvc_candidates(conda_prefix: str | None) -> list[dict[str, Any]]:
    """MSVC runtime copies inside the environment, paired with the System32 one."""
    directories: list[tuple[str, str]] = [
        (os.path.dirname(sys.executable), SOURCE_PYTHON_DIR)
    ]
    if conda_prefix:
        directories.append(
            (os.path.join(conda_prefix, "Library", "bin"), SOURCE_CONDA_LIBRARY_BIN)
        )
    directories.append((_system32(), SOURCE_SYSTEM32))

    found: list[dict[str, Any]] = []
    for name in MSVC_RUNTIME_DLLS:
        for directory, source in directories:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            found.append({"name": name, "path": path, "source": source,
                          "version": dll_file_version(path)})
    return found


def qt_environment() -> dict[str, Any]:
    """Everything :func:`diagnose` needs, gathered from the live environment.

    Split from :func:`diagnose` so the rules are a pure function over plain data and
    can be tested without a Conda install, a Windows box, or a broken Qt.
    """
    conda_prefix = _conda_prefix()
    package_dir = pyqt6_package_dir()
    wheel_bin = wheel_qt_bin_dir(package_dir)
    wheel_dll = os.path.join(wheel_bin, QT_CORE_DLL) if wheel_bin else None
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "conda_prefix": conda_prefix,
        "distributions": {
            BINDING_DIST: _distribution_version(BINDING_DIST),
            RUNTIME_DIST: _distribution_version(RUNTIME_DIST),
            SIP_DIST: _distribution_version(SIP_DIST),
        },
        "package_dir": package_dir,
        "wheel_qt_bin": wheel_bin,
        # The wheel's own Qt6Core.dll version, read from the file. Preferred over the
        # PyQt6-Qt6 metadata version because it is what the binding was built against
        # and it exists even when the metadata does not.
        "wheel_qt_version": (
            dll_file_version(wheel_dll)
            if wheel_dll and os.path.isfile(wheel_dll) else None
        ),
        "qt_core_candidates": _qt_core_candidates(conda_prefix, package_dir, wheel_bin),
        "msvc_candidates": _msvc_candidates(conda_prefix),
    }


# --- the rules -------------------------------------------------------------


_CLEAN_ENV_REMEDY = (
    "Install into a clean virtual environment that has no Qt of its own:\n"
    "    python -m venv .venv\n"
    "    .venv\\Scripts\\activate\n"
    "    pip install -e ."
)


def diagnose(env: dict[str, Any], *, qt_failed: bool = False) -> list[Finding]:
    """Rules over the :func:`qt_environment` mapping. Pure; no I/O.

    ``qt_failed`` says whether Qt has *actually* failed to import. It gates the rules
    whose evidence is only meaningful in that context — see :func:`_diagnose_msvc_shadowing`.
    ``sreeni-cli doctor`` runs without it (a proactive preflight must not cry wolf);
    :func:`format_import_failure` passes it, because there the failure is a given and
    the question is only what explains it.
    """
    findings: list[Finding] = []
    distributions = env.get("distributions", {})
    binding = distributions.get(BINDING_DIST)

    if binding is None:
        return [Finding(
            severity=SEVERITY_ERROR,
            title="PyQt6 is not installed",
            detail="No PyQt6 distribution metadata was found for this interpreter.",
            remedy="pip install -e .   (in the environment you actually run the app from)",
        )]

    findings.extend(_diagnose_pip_skew(env, binding))

    # Everything below reasons about a file called Qt6Core.dll and about how Windows
    # resolves it. On Linux that file is libQt6Core.so.6 and on macOS it lives inside
    # QtCore.framework, so the probe finds nothing -- and every rule below would read
    # that absence as breakage, on machines where PyQt6 demonstrably imports.
    #
    # One gate, here, rather than a platform check inside each rule: the same bug has
    # already been shipped twice from two different functions (an error on a missing
    # PyQt6-Qt6 distribution, then an error on an empty candidate list), both times
    # recommending a force-reinstall to someone whose install was fine. The invariant
    # is "this rule set only makes claims about Windows", and it belongs in one place.
    if env.get("platform") != "win32":
        return findings

    findings.extend(_diagnose_qt_resolution(env, binding))
    if qt_failed:
        findings.extend(_diagnose_msvc_shadowing(env))
    return findings


def _diagnose_pip_skew(env: dict[str, Any], binding: str) -> list[Finding]:
    """Binding and Qt-runtime wheels at different minors.

    Only meaningful when the Qt runtime came from the ``PyQt6-Qt6`` wheel. A PyQt6
    built against a system Qt has no such distribution, and that is a healthy install
    -- Linux distro packages, Homebrew and ``conda install pyqt`` all look like this.
    Erroring on them would fail `doctor` on working machines and, worse, the obvious
    remedy (force-reinstall pip's PyQt6 over a conda-managed one) is how you
    manufacture issue #92 on a machine that did not have it.
    """
    runtime = (env.get("distributions") or {}).get(RUNTIME_DIST)
    if runtime is None:
        return []
    binding_mm, runtime_mm = _major_minor(binding), _major_minor(runtime)
    if not binding_mm or not runtime_mm or binding_mm == runtime_mm:
        return []
    return [Finding(
        severity=SEVERITY_ERROR,
        title="PyQt6 and its Qt runtime wheel are different minor versions",
        detail=f"{BINDING_DIST} {binding} against {RUNTIME_DIST} {runtime}.",
        remedy=f'pip install --force-reinstall "{BINDING_DIST}=={binding}"',
    )]


def _expected_qt(env: dict[str, Any]) -> tuple[tuple[int, int] | None, str]:
    """The Qt version PyQt6 expects, and where that number came from.

    The wheel's own ``Qt6Core.dll`` is preferred over ``PyQt6-Qt6`` metadata: it is
    what the binding was compiled against, and it is readable even when the metadata
    is absent. When neither is available the caller falls back to the binding's own
    version, since PyQt6 6.11 is built against Qt 6.11.x.
    """
    wheel_version = _major_minor(env.get("wheel_qt_version"))
    if wheel_version:
        return wheel_version, "the Qt it ships"
    runtime = _major_minor((env.get("distributions") or {}).get(RUNTIME_DIST))
    if runtime:
        return runtime, f"{RUNTIME_DIST}"
    return None, ""


def _diagnose_qt_resolution(env: dict[str, Any], binding: str) -> list[Finding]:
    """Which ``Qt6Core.dll`` actually wins, and whether that is the right one."""
    candidates = env.get("qt_core_candidates") or []
    if not candidates:
        return [Finding(
            severity=SEVERITY_ERROR,
            title=f"No {QT_CORE_DLL} found anywhere PyQt6 will look",
            detail=(
                f"{BINDING_DIST} {binding} is installed but no Qt runtime was found in "
                "the interpreter's directory, the PyQt6 package, or PATH."
            ),
            remedy=f'pip install --force-reinstall "{BINDING_DIST}=={binding}"',
        )]

    effective = candidates[0]
    if effective["source"] == SOURCE_WHEEL:
        return []

    found = _major_minor(effective["version"])
    wheel_bin = env.get("wheel_qt_bin")

    expected, expected_from = _expected_qt(env)
    if not expected:
        # No Qt in the wheel and no runtime metadata: this copy is simply the Qt in
        # use, so judge it against the binding's own version instead.
        expected, expected_from = _major_minor(binding), BINDING_DIST
    # Only claim something is being *shadowed* when there is a Qt in the wheel to
    # shadow. The directory existing is not enough: a partial or corrupted
    # PyQt6-Qt6 leaves an empty Qt6/bin, and that absence is precisely why something
    # else won -- saying it was preferred over the wheel's Qt would invert the story.
    shadowed = (
        f"the Qt PyQt6 ships in {wheel_bin}"
        if wheel_bin and env.get("wheel_qt_version") is not None else None
    )

    location = f"{effective['path']} (version {effective['version'] or 'unreadable'})"

    if expected and found and expected != found:
        return [_mismatch_finding(effective, location, shadowed, expected,
                                  expected_from, found)]
    if expected and found:
        if shadowed is None:
            return []
        return [Finding(
            severity=SEVERITY_WARNING,
            title="Another Qt runtime is used instead of the one PyQt6 ships",
            detail=(
                f"{location} is used in preference to {shadowed}. Both are "
                f"{found[0]}.{found[1]}.x, so nothing is broken today -- but upgrading "
                "PyQt6 would make them diverge, which is exactly how issue #92 happened."
            ),
            remedy=_CLEAN_ENV_REMEDY,
        )]

    # One of the two numbers could not be read, so no claim can be made either way.
    # Saying "the versions match" here would be asserting something unknown.
    return [Finding(
        severity=SEVERITY_SUSPECT,
        title="A Qt runtime outside PyQt6 is being used",
        detail=(
            f"{location} takes precedence"
            + (f" over {shadowed}" if shadowed else "")
            + ". Its version could not be determined, so whether it matches what PyQt6 "
            "needs is unknown -- but a foreign Qt is the usual cause of this failure."
        ),
        remedy=_CLEAN_ENV_REMEDY,
    )]


def _mismatch_finding(effective: dict[str, Any], location: str, shadowed: str | None,
                      expected: tuple[int, int], expected_from: str,
                      found: tuple[int, int]) -> Finding:
    remedy = _CLEAN_ENV_REMEDY
    if effective["source"] == SOURCE_PACKAGE_DIR:
        # A Qt6Core.dll sitting next to QtCore.pyd was put there by hand -- it is the
        # fix people try for this error before they know the cause. Deleting it is
        # unambiguously right, and telling them to pin the binding to match a DLL they
        # dropped there themselves would be absurd.
        remedy = (
            f"Delete {effective['path']} -- it was placed inside the PyQt6 package by\n"
            "hand and overrides the Qt the wheel ships.\n"
            f"  or, to start clean:\n{_CLEAN_ENV_REMEDY}"
        )
        return Finding(
            severity=SEVERITY_ERROR,
            title="A hand-placed Qt6Core.dll inside the PyQt6 package overrides its own Qt",
            detail=(
                f"{location}. PyQt6 expects Qt {expected[0]}.{expected[1]}.x (from "
                f"{expected_from}) but that copy is {found[0]}.{found[1]}.x."
            ),
            remedy=remedy,
        )
    if effective["source"] in (SOURCE_CONDA_LIBRARY_BIN, SOURCE_PATH, SOURCE_PYTHON_DIR):
        remedy += (
            "\n  or remove the conflicting Qt if nothing in the environment needs it:\n"
            "    conda remove qt6-main"
        )
    # Only offer "downgrade the binding to match" when the file really is a Qt of the
    # same major. Anything else is a foreign DLL wearing the name, and emitting
    # `pip install "PyQt6==10.0.*"` would hand the user a command that cannot resolve.
    if found[0] == expected[0]:
        remedy += (
            "\n  or match the binding to the Qt already present:\n"
            f'    pip install "PyQt6=={found[0]}.{found[1]}.*"'
        )
    return Finding(
        severity=SEVERITY_ERROR,
        title="A different Qt runtime shadows the one PyQt6 needs",
        detail=(
            f"{location} is used"
            + (f" in preference to {shadowed}" if shadowed else "")
            + f". PyQt6 expects Qt {expected[0]}.{expected[1]}.x (from {expected_from}) "
            f"but that copy is {found[0]}.{found[1]}.x, so QtCore resolves against a Qt "
            "that does not export the symbols it needs -- the 0xc0000139 crash."
        ),
        remedy=remedy,
    )


def _diagnose_msvc_shadowing(env: dict[str, Any]) -> list[Finding]:
    """Flag an environment-local MSVC runtime older than the system's.

    **Only runs when Qt has actually failed to import** (``qt_failed``), and this is
    the whole reason that parameter exists. CPython's own Windows installer bundles
    the VC redistributable next to ``python.exe``, and it is routinely a minor behind
    System32's: a stock GitHub Actions runner has 14.42.34438.0 beside the interpreter
    and 14.51.36247.0 in System32. As an unconditional rule this therefore fires on
    essentially *every* Windows Python — it failed the whole windows-latest CI matrix,
    which is how it was caught, rather than on any machine with a real problem.

    Narrowing the comparison does not rescue it: 14.42 vs 14.51 is a genuine minor
    difference, and the 14.x redistributable is deliberately binary-compatible across
    exactly that range, so the version gap carries almost no signal on its own.

    What it does carry is *conditional* signal. Once Qt has demonstrably failed with
    the mismatch signature, an older bundled runtime is a plausible explanation worth
    putting in front of the user — as a ``suspect``, since confirming it would mean
    loading the DLL and reading its export table, which this module will not do.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for candidate in env.get("msvc_candidates", []):
        by_name.setdefault(candidate["name"], []).append(candidate)

    findings: list[Finding] = []
    for name, candidates in by_name.items():
        system = next((c for c in candidates if c["source"] == SOURCE_SYSTEM32), None)
        local = [c for c in candidates if c["source"] != SOURCE_SYSTEM32]
        if system is None or not local:
            continue
        # Compared at major.minor, NOT the full four-part version. MSVC STL symbol
        # additions land on the minor (14.29 -> 14.42); the build number tracks
        # servicing. Conda's vc14_runtime trails Windows Update's build essentially
        # always, so comparing all four parts would fire this -- which now fails
        # `doctor` -- on a large share of perfectly healthy conda installs.
        system_version = _major_minor(system["version"])
        for candidate in local:
            local_version = _major_minor(candidate["version"])
            if not system_version or not local_version or local_version >= system_version:
                continue
            findings.append(Finding(
                severity=SEVERITY_SUSPECT,
                title=f"An older {name} shadows the system one",
                detail=(
                    f"{candidate['path']} is version {candidate['version']} while "
                    f"{system['path']} is {system['version']}. Qt is built against the "
                    "newer MSVC runtime and may import symbols the older copy lacks, "
                    "which produces the same 0xc0000139 from a different DLL. Note that "
                    "unlike the Qt case this directory is not necessarily on the search "
                    "path -- msvcp140.dll is resolved by the loader, not by PyQt6's "
                    "find_qt(), so this copy only wins if something put it there."
                ),
                remedy=(
                    "Update the environment's runtime (conda update vc14_runtime) or "
                    f"delete {candidate['path']} so the system copy is used."
                ),
            ))
    return findings


# --- rendering -------------------------------------------------------------


def format_report(env: dict[str, Any], findings: list[Finding]) -> str:
    """Full environment report for ``sreeni-cli doctor``."""
    lines: list[str] = ["Qt environment", "--------------"]
    lines.append(f"  platform     {env.get('platform')}")
    lines.append(f"  python       {env.get('python')}")
    lines.append(f"  interpreter  {env.get('executable')}")
    lines.append(f"  conda        {env.get('conda_prefix') or 'not detected'}")

    lines.append("")
    lines.append("Distributions")
    for name, value in (env.get("distributions") or {}).items():
        lines.append(f"  {name:<12} {value or 'not installed'}")
    # "no bundled Qt" and "could not read its version" are different states, and only
    # the directory check distinguishes them: wheel_qt_version comes from the PE
    # resource reader, which returns None on every non-Windows platform. Rendering the
    # falsy value as a conclusion would assert "PyQt6 uses a system Qt" about a plain
    # pip install on Linux that ships the PyQt6-Qt6 wheel.
    wheel_bin = env.get("wheel_qt_bin")
    if wheel_bin:
        if env.get("platform") == "win32":
            detail = env.get("wheel_qt_version") or "version unreadable"
        else:
            detail = "present (version readable on Windows only)"
        lines.append(f"  wheel Qt     {detail}")
        lines.append(f"               {wheel_bin}")
    else:
        lines.append("  wheel Qt     none bundled (PyQt6 uses a system Qt)")

    # Off Windows both this section and the wheel-Qt line above are empty by
    # construction -- the probe looks for a filename that only exists there. Say so,
    # or a Linux user pastes "none found" into a bug report as if it were the symptom.
    windows_only = "" if env.get("platform") == "win32" else "   (Windows only)"
    lines.append("")
    lines.append(
        f"{QT_CORE_DLL}, in the order PyQt6 consults it (first one wins){windows_only}"
    )
    candidates = env.get("qt_core_candidates") or []
    if candidates:
        for index, candidate in enumerate(candidates, start=1):
            marker = " <- used" if index == 1 else ""
            version = candidate.get("version") or "unreadable"
            lines.append(f"  {index}. [{candidate['source']}] {version}{marker}")
            lines.append(f"     {candidate['path']}")
    else:
        lines.append("  none found")

    # Printed unconditionally: a bug report pasted from this command should carry the
    # MSVC picture whether or not a rule happened to fire on it.
    lines.append("")
    lines.append(f"MSVC runtime{windows_only}")
    msvc = env.get("msvc_candidates") or []
    if msvc:
        for candidate in msvc:
            lines.append(
                f"  {candidate['name']:<22} {candidate.get('version') or 'unreadable':<18}"
                f"[{candidate['source']}]"
            )
    else:
        lines.append("  none found (normal off Windows)")

    lines.append("")
    if not findings:
        lines.append("No problems detected.")
        return "\n".join(lines)

    lines.append(f"Findings ({len(findings)})")
    lines.append("--------")
    lines.extend(_format_findings(findings))
    return "\n".join(lines)


def _format_findings(findings: list[Finding]) -> list[str]:
    lines: list[str] = []
    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. [{finding.severity}] {finding.title}")
        lines.append(f"   {finding.detail}")
        for remedy_line in finding.remedy.splitlines():
            lines.append(f"   {remedy_line}")
        lines.append("")
    return lines


def format_import_failure(exc: BaseException) -> str:
    """Message shown when the GUI entry point cannot import Qt.

    Never raises: this runs inside an exception handler, and a traceback from the
    diagnostic would bury the original failure it is meant to explain.
    """
    lines = [
        "ERROR: the Qt runtime (PyQt6) failed to load, so the application cannot start.",
        "",
        f"  {type(exc).__name__}: {exc}",
        "",
    ]
    try:
        env = qt_environment()
        # qt_failed=True: Qt demonstrably did not load, so evidence that would be too
        # weak to raise proactively is worth showing here.
        findings = diagnose(env, qt_failed=True)
        if findings:
            lines.append("Likely cause")
            lines.append("------------")
            lines.extend(_format_findings(findings))
        else:
            lines.append(
                "No conflicting Qt installation was detected, so this is not the known"
            )
            lines.append("Conda shadowing problem. Full environment detail:")
            lines.append("")
            lines.append(format_report(env, findings))
            lines.append("")
    except Exception:  # pragma: no cover - the diagnostic must never mask the error
        lines.append("(the environment diagnostic itself failed to run)")
        lines.append("")

    lines.append("Run 'sreeni-cli doctor' for the full report -- it does not load Qt, so it")
    lines.append("still works in an environment where this application cannot start.")
    lines.append("See the Troubleshooting section of README.md.")
    return "\n".join(lines)
