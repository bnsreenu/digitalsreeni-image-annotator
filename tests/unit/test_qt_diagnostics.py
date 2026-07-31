"""Rules and probe behind `sreeni-cli doctor` and the main.py Qt guard (issue #92).

`diagnose` is a pure function over the mapping `qt_environment` produces, which is the
whole reason the two are separate: the failure being diagnosed needs a Windows box, a
Conda install and a *broken* Qt to reproduce, and none of those exist in CI. Feeding it
hand-built environments tests the rules everywhere.

The probe (`_qt_core_candidates`) is tested too, with a fake environment on disk. Every
ordering claim this feature makes is a claim about that function, so leaving it to a
key-existence smoke check would mean the diagnosis itself was the untested part.
"""

import os
import sys

import pytest

from src.digitalsreeni_image_annotator.core import qt_diagnostics as qd

WHEEL_BIN = r"C:\envs\ann\Lib\site-packages\PyQt6\Qt6\bin"


def _dll(source, version, path=None):
    return {
        "path": path or os.path.join("C:\\", source, qd.QT_CORE_DLL),
        "source": source,
        "version": version,
    }


def _env(binding="6.11.0", runtime="6.11.1", sip="13.10.2", qt_core=None, msvc=None,
         conda_prefix=None, wheel_bin=WHEEL_BIN, wheel_qt_version="6.11.1.0"):
    """An environment mapping shaped exactly like `qt_environment()` returns."""
    return {
        "platform": "win32",
        "python": "3.10.13",
        "executable": r"C:\envs\ann\python.exe",
        "prefix": r"C:\envs\ann",
        "conda_prefix": conda_prefix,
        "distributions": {
            qd.BINDING_DIST: binding,
            qd.RUNTIME_DIST: runtime,
            qd.SIP_DIST: sip,
        },
        "package_dir": r"C:\envs\ann\Lib\site-packages\PyQt6",
        "wheel_qt_bin": wheel_bin,
        "wheel_qt_version": wheel_qt_version,
        "qt_core_candidates": qt_core if qt_core is not None else [
            _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
        ],
        "msvc_candidates": msvc or [],
    }


def _severities(findings):
    return [finding.severity for finding in findings]


# --- healthy environments must stay silent ---------------------------------


def test_a_wheel_only_environment_is_clean():
    assert qd.diagnose(_env()) == []


def test_a_qt_after_the_wheel_is_not_a_problem():
    """Only the first entry is ever loaded, so later ones are noise. A diagnostic that
    cries wolf on every healthy Windows box is worse than none."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
        _dll(qd.SOURCE_SYSTEM32, "6.2.0.0"),
    ])
    assert qd.diagnose(env) == []


def test_a_patch_level_difference_is_normal():
    """PyQt6 6.11.0 ships against PyQt6-Qt6 6.11.1 -- flagging that would fire on every
    correct install."""
    assert qd.diagnose(_env(binding="6.11.0", runtime="6.11.1")) == []


def test_a_system_qt_install_without_the_runtime_wheel_is_healthy():
    """Linux distro packages, Homebrew and `conda install pyqt` all ship PyQt6 built
    against a system Qt: no PyQt6-Qt6 distribution, no bundled Qt6Core.dll. Erroring
    here would fail `doctor` on working machines -- and its obvious remedy
    (force-reinstall pip's PyQt6 over the conda-managed one) is how you *manufacture*
    issue #92 on a machine that did not have it."""
    env = _env(
        binding="6.8.0", runtime=None, wheel_bin=None, wheel_qt_version=None,
        qt_core=[_dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0")],
        conda_prefix=r"C:\envs\ann",
    )
    assert qd.diagnose(env) == []


# --- the reported failure --------------------------------------------------


def test_an_older_conda_qt_ahead_of_the_wheel_is_an_error():
    """The issue #92 scenario: conda-forge's Qt 6.8 wins over a PyQt6 6.11 wheel."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ], conda_prefix=r"C:\envs\ann")
    findings = qd.diagnose(env)

    assert _severities(findings) == [qd.SEVERITY_ERROR]
    finding = findings[0]
    assert "6.11" in finding.detail and "6.8" in finding.detail
    assert qd.SOURCE_CONDA_LIBRARY_BIN in finding.detail
    assert "python -m venv" in finding.remedy
    assert "conda remove" in finding.remedy
    assert 'pip install "PyQt6==6.8.*"' in finding.remedy


def test_the_interpreter_directory_wins_outright():
    """find_qt() checks dirname(sys.executable) first and, when Qt is there, never
    registers the wheel's at all. For a Conda env that directory IS the env root."""
    env = _env(qt_core=[_dll(qd.SOURCE_PYTHON_DIR, "6.8.2.0")])
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert qd.SOURCE_PYTHON_DIR in findings[0].detail


def test_the_expected_version_comes_from_the_wheel_dll_not_only_metadata():
    """PyQt6-Qt6 metadata can be absent while the wheel's Qt6Core.dll is right there.
    Falling back to 'unknown' in that case would downgrade the #92 error to a shrug."""
    env = _env(
        runtime=None, wheel_qt_version="6.11.1.0",
        qt_core=[
            _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
            _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
        ],
    )
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "the Qt it ships" in findings[0].detail


def test_a_hand_placed_dll_inside_the_package_is_told_to_delete_it():
    """A Qt6Core.dll next to QtCore.pyd was put there by hand -- it is the fix people
    try for this error before they know the cause. "Pin the binding to match the DLL
    you dropped there yourself" would be absurd advice."""
    # Both entries: the probe always continues past the package dir to the wheel, so
    # a package-dir-only list is a state it cannot emit.
    env = _env(qt_core=[
        _dll(qd.SOURCE_PACKAGE_DIR, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)

    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "Delete" in findings[0].remedy
    assert "pip install \"PyQt6==" not in findings[0].remedy
    assert "hand" in findings[0].title


def test_a_foreign_dll_wearing_the_name_gets_no_pip_pin_advice():
    """Whatever this is, it is not a Qt 6 build -- so `pip install "PyQt6==10.0.*"`
    would be a command that cannot resolve. Removing it is the only honest advice."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_PYTHON_DIR, "10.0.26100.8875"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "PyQt6==" not in findings[0].remedy
    assert "python -m venv" in findings[0].remedy


def test_an_empty_wheel_qt_directory_is_not_described_as_shadowed():
    """A partial or corrupted PyQt6-Qt6 leaves Qt6/bin present but empty. That absence
    is *why* something else won, so reporting the other copy as "used in preference to
    the Qt PyQt6 ships in <dir>" would invert the story."""
    env = _env(wheel_qt_version=None, runtime="6.11.1",
               qt_core=[_dll(qd.SOURCE_SYSTEM32, "6.2.0.0")])
    findings = qd.diagnose(env)

    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "in preference to" not in findings[0].detail
    assert "6.11" in findings[0].detail  # still judged against the metadata version


def test_a_binding_newer_than_the_system_qt_is_an_error():
    """No wheel Qt to shadow, so the binding itself is what the system Qt must match."""
    env = _env(binding="6.11.0", runtime=None, wheel_bin=None, wheel_qt_version=None,
               qt_core=[_dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0")])
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert qd.BINDING_DIST in findings[0].detail


# --- "we cannot tell" must never be reported as "it matches" ---------------


def test_an_unreadable_shadow_version_is_a_suspect_not_a_match_claim():
    """Regression: the warning branch used to fire whenever EITHER version was
    unparsable while its text said "Its version matches, so it is not currently
    breaking the import" -- i.e. it printed two mismatched numbers and then asserted
    they agreed, downgrading the #92 case to a warning whenever PyQt6-Qt6 metadata was
    missing or the shadow had no version resource."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, None),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)

    assert _severities(findings) == [qd.SEVERITY_SUSPECT]
    assert "could not be determined" in findings[0].detail
    # The exact false claim that used to appear here. "whether it matches ... is
    # unknown" is fine; asserting agreement is not.
    assert "version matches" not in findings[0].detail
    assert "not currently breaking" not in findings[0].detail


def test_the_binding_version_is_the_last_resort_expectation():
    """With the wheel's Qt unreadable AND no runtime metadata there is still a third
    signal: PyQt6 6.11 is built against Qt 6.11.x, so the binding's own version says
    what is expected. Falling back to "unknown" here would downgrade a provable
    mismatch to a shrug."""
    env = _env(binding="6.11.0", runtime=None, wheel_qt_version=None, qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, None),
    ])
    findings = qd.diagnose(env)

    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert qd.BINDING_DIST in findings[0].detail
    assert "version matches" not in findings[0].detail
    assert "not currently breaking" not in findings[0].detail


def test_a_genuinely_matching_shadow_warns_and_says_so():
    """Same version, so nothing is broken today -- but the next PyQt6 upgrade breaks it,
    which is precisely how the reporter got there."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.11.1.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_WARNING]
    assert "6.11" in findings[0].detail
    assert "upgrading" in findings[0].detail


def test_only_errors_and_suspects_fail_the_doctor_command():
    """A warning is a forecast; it must not fail a build. A suspect could be the cause
    of a GUI that will not start, so it must."""
    assert qd.SEVERITY_ERROR in qd.FAILING_SEVERITIES
    assert qd.SEVERITY_SUSPECT in qd.FAILING_SEVERITIES
    assert qd.SEVERITY_WARNING not in qd.FAILING_SEVERITIES


# --- pip-side skew and absent installs -------------------------------------


def test_missing_pyqt6_reports_that_and_stops():
    findings = qd.diagnose(_env(binding=None))
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "not installed" in findings[0].title


def test_binding_and_runtime_minor_skew_is_an_error():
    findings = qd.diagnose(_env(
        binding="6.11.0", runtime="6.8.2", wheel_qt_version="6.8.2.0",
        qt_core=[_dll(qd.SOURCE_WHEEL, "6.8.2.0")],
    ))
    assert any("different minor versions" in f.title for f in findings)


def test_no_qt_anywhere_on_windows_is_an_error():
    env = _env(qt_core=[], wheel_bin=None, wheel_qt_version=None, runtime=None)
    findings = qd.diagnose(env)
    assert _severities(findings) == [qd.SEVERITY_ERROR]
    assert "No Qt6Core.dll" in findings[0].title


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_dll_rules_make_no_claims_off_windows(platform):
    """On Linux the file is libQt6Core.so.6 and on macOS it is inside
    QtCore.framework, so the probe -- which looks for the literal name Qt6Core.dll --
    finds nothing. Every rule that reasons about that name would read the absence as
    breakage and tell a user whose install works fine to force-reinstall PyQt6.

    This shipped twice from two different rules before the platform gate went in one
    place, which is why it is asserted here rather than left to the live check below.
    """
    env = _env(qt_core=[], wheel_bin=None, wheel_qt_version=None, runtime=None,
               msvc=[])
    env["platform"] = platform
    assert qd.diagnose(env) == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_pip_skew_is_still_reported_off_windows(platform):
    """The gate covers the DLL rules, not everything: a binding/runtime wheel skew is
    read from package metadata and is just as wrong on Linux."""
    env = _env(binding="6.11.0", runtime="6.8.2", qt_core=[], wheel_bin=None,
               wheel_qt_version=None)
    env["platform"] = platform
    assert any("different minor versions" in f.title for f in qd.diagnose(env))


# --- the second shadowing path: the MSVC runtime ---------------------------


def _msvc(source, version, name="msvcp140.dll"):
    return {
        "name": name,
        "path": os.path.join("C:\\", source, name),
        "source": source,
        "version": version,
    }


def test_a_newer_system_build_number_is_not_treated_as_a_mismatch():
    """Conda's vc14_runtime trails Windows Update's servicing builds essentially
    always. MSVC STL symbol additions land on the minor (14.29 -> 14.42), not the
    build, so comparing all four parts would fail `doctor` on a large share of healthy
    conda installs -- on the exact platform this tool targets."""
    env = _env(msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.44.35208.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.44.35211.0"),
    ])
    assert qd.diagnose(env, qt_failed=True) == []


def test_a_stock_cpython_layout_is_not_flagged_by_a_proactive_doctor_run():
    """CPython's Windows installer bundles the VC redistributable next to python.exe,
    routinely a minor behind System32's: a stock GitHub Actions runner has
    14.42.34438.0 beside the interpreter and 14.51.36247.0 in System32. As an
    unconditional rule this fired on EVERY windows-latest CI leg -- 100% false
    positives on healthy machines -- which is what `qt_failed` exists to fix."""
    env = _env(msvc=[
        _msvc(qd.SOURCE_PYTHON_DIR, "14.42.34438.0", name="vcruntime140.dll"),
        _msvc(qd.SOURCE_SYSTEM32, "14.51.36247.0", name="vcruntime140.dll"),
    ])
    assert qd.diagnose(env) == []


def test_the_same_layout_is_offered_as_evidence_once_qt_has_failed():
    """The version gap carries no signal on its own, but it is a plausible explanation
    once Qt has demonstrably failed with the mismatch signature."""
    env = _env(msvc=[
        _msvc(qd.SOURCE_PYTHON_DIR, "14.42.34438.0", name="vcruntime140.dll"),
        _msvc(qd.SOURCE_SYSTEM32, "14.51.36247.0", name="vcruntime140.dll"),
    ])
    findings = qd.diagnose(env, qt_failed=True)
    assert _severities(findings) == [qd.SEVERITY_SUSPECT]


def test_an_older_local_msvc_runtime_is_a_suspect():
    """The module docstring bills this as one of the two mechanisms that produce the
    fatal 0xc0000139, so it has to fail `doctor` -- a user whose app cannot start must
    not get a clean exit code from the tool built to explain why. It stays a suspect
    rather than an error because confirming it would mean loading the DLL."""
    env = _env(msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.29.30139.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0"),
    ])
    findings = qd.diagnose(env, qt_failed=True)
    assert _severities(findings) == [qd.SEVERITY_SUSPECT]
    assert findings[0].severity in qd.FAILING_SEVERITIES
    assert "msvcp140.dll" in findings[0].title


def test_a_newer_or_equal_local_msvc_runtime_is_fine():
    env = _env(msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.42.34433.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0"),
    ])
    assert qd.diagnose(env, qt_failed=True) == []


def test_msvc_without_a_system_copy_is_not_judged():
    env = _env(msvc=[_msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.29.30139.0")])
    assert qd.diagnose(env, qt_failed=True) == []


def test_vcruntime140_1_is_watched_too():
    """It hosts __CxxFrameHandler4 and is the classic STATUS_ENTRYPOINT_NOT_FOUND
    culprit in a mixed-toolchain environment."""
    assert "vcruntime140_1.dll" in qd.MSVC_RUNTIME_DLLS


# --- the probe: what the loader will actually consult ----------------------


def _make_dll(directory, name=qd.QT_CORE_DLL):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_bytes(b"not a real PE")
    return target


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """An interpreter, a PyQt6 package and a PATH, all under tmp_path."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "env" / "python.exe"))
    (tmp_path / "env").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "windows"))
    return tmp_path


def test_the_interpreter_directory_short_circuits_the_wheel(fake_env):
    """find_qt() returns as soon as it finds Qt next to python.exe, so the wheel's own
    Qt is never registered -- it must not appear in the list at all."""
    _make_dll(fake_env / "env")
    wheel_bin = fake_env / "site-packages" / "PyQt6" / "Qt6" / "bin"
    _make_dll(wheel_bin)

    found = qd._qt_core_candidates(None, str(fake_env / "site-packages" / "PyQt6"),
                                   str(wheel_bin))

    assert [c["source"] for c in found] == [qd.SOURCE_PYTHON_DIR]


def test_the_wheel_wins_when_the_interpreter_directory_is_clean(fake_env):
    wheel_bin = fake_env / "site-packages" / "PyQt6" / "Qt6" / "bin"
    _make_dll(wheel_bin)

    found = qd._qt_core_candidates(None, str(fake_env / "site-packages" / "PyQt6"),
                                   str(wheel_bin))

    assert [c["source"] for c in found] == [qd.SOURCE_WHEEL]


def test_the_package_directory_outranks_everything(fake_env):
    """LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR puts the directory holding QtCore.pyd ahead of
    anything find_qt registers, and dropping a DLL there is a common (wrong) fix
    people try for this very error."""
    package_dir = fake_env / "site-packages" / "PyQt6"
    _make_dll(package_dir)
    wheel_bin = package_dir / "Qt6" / "bin"
    _make_dll(wheel_bin)

    found = qd._qt_core_candidates(None, str(package_dir), str(wheel_bin))

    assert found[0]["source"] == qd.SOURCE_PACKAGE_DIR


def test_path_is_walked_when_the_wheel_ships_no_qt(fake_env, monkeypatch):
    """CPython >= 3.8 ignores PATH for extension-module dependencies, so it looks like
    PATH cannot matter -- but PyQt6's find_qt() puts it back, walking PATH and
    add_dll_directory-ing the first entry with a Qt6Core.dll. That is how a conda
    env's Library\\bin becomes the registered directory."""
    empty = fake_env / "nothing"
    empty.mkdir()
    conda_prefix = fake_env / "conda"
    library_bin = conda_prefix / "Library" / "bin"
    _make_dll(library_bin)
    monkeypatch.setenv("PATH", os.pathsep.join([str(empty), str(library_bin)]))

    found = qd._qt_core_candidates(str(conda_prefix), None, None)

    assert [c["source"] for c in found] == [qd.SOURCE_CONDA_LIBRARY_BIN]
    assert str(library_bin) in found[0]["path"]


def test_a_non_conda_path_entry_is_labelled_generically(fake_env, monkeypatch):
    stray = fake_env / "some" / "other" / "qt"
    _make_dll(stray)
    monkeypatch.setenv("PATH", str(stray))

    found = qd._qt_core_candidates(None, None, None)

    assert [c["source"] for c in found] == [qd.SOURCE_PATH]


def test_only_the_first_matching_path_entry_is_taken(fake_env, monkeypatch):
    """find_qt breaks out of its PATH loop on the first hit, so listing later ones
    would imply a fallback the loader never performs."""
    first = fake_env / "first"
    second = fake_env / "second"
    _make_dll(first)
    _make_dll(second)
    monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second)]))

    found = qd._qt_core_candidates(None, None, None)

    assert len(found) == 1
    assert str(first) in found[0]["path"]


def test_nothing_anywhere_yields_no_candidates(fake_env):
    assert qd._qt_core_candidates(None, None, None) == []


def test_conda_prefix_is_detected_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert qd._conda_prefix() == str(tmp_path)


def test_conda_prefix_is_detected_from_conda_meta_without_activation(tmp_path,
                                                                    monkeypatch):
    """The app is often launched from a shortcut or an IDE that never ran
    `conda activate`, so CONDA_PREFIX is unset."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    (tmp_path / "conda-meta").mkdir()
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert qd._conda_prefix() == str(tmp_path)


def test_no_conda_is_reported_as_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert qd._conda_prefix() is None


def test_msvc_probe_pairs_the_environment_copy_with_the_system_one(fake_env):
    conda_prefix = fake_env / "conda"
    _make_dll(conda_prefix / "Library" / "bin", "msvcp140.dll")
    _make_dll(fake_env / "windows" / "System32", "msvcp140.dll")

    found = qd._msvc_candidates(str(conda_prefix))
    sources = [c["source"] for c in found if c["name"] == "msvcp140.dll"]

    assert qd.SOURCE_CONDA_LIBRARY_BIN in sources
    assert qd.SOURCE_SYSTEM32 in sources


# --- version parsing -------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("6.11.1.0", (6, 11)),
    ("6.8.2", (6, 8)),
    ("6.11", (6, 11)),
    ("6", None),
    ("", None),
    (None, None),
    ("6.x.1", None),
])
def test_major_minor_parsing(value, expected):
    assert qd._major_minor(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("14.42.34433.0", (14, 42, 34433, 0)),
    ("1.2", (1, 2)),
    ("1.2.beta", None),
    (None, None),
])
def test_version_tuple_parsing(value, expected):
    assert qd._version_tuple(value) == expected


# --- rendering -------------------------------------------------------------


def _broken_env():
    return _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ], conda_prefix=r"C:\envs\ann", msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.29.30139.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0"),
    ])


def test_the_report_is_ascii_only():
    """A Windows console under a legacy code page mangles non-ASCII exactly when the
    output is redirected, which is what pasting a bug report does."""
    env = _broken_env()
    qd.format_report(env, qd.diagnose(env)).encode("ascii")


def test_the_report_names_every_candidate_in_search_order():
    env = _broken_env()
    report = qd.format_report(env, qd.diagnose(env))
    assert report.index(qd.SOURCE_CONDA_LIBRARY_BIN) < report.index(qd.SOURCE_WHEEL)
    assert "6.8.2.0" in report and "6.11.1.0" in report
    assert "<- used" in report


def test_the_report_always_carries_the_msvc_picture():
    """A bug report pasted from `doctor` should contain the MSVC state whether or not
    a rule happened to fire on it."""
    clean = _env(msvc=[_msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0")])
    report = qd.format_report(clean, qd.diagnose(clean))
    assert "MSVC runtime" in report
    assert "14.42.34433.0" in report
    assert "No problems detected." in report


def test_the_report_states_when_pyqt6_ships_no_qt():
    env = _env(wheel_bin=None, wheel_qt_version=None, runtime=None,
               qt_core=[_dll(qd.SOURCE_SYSTEM32, "6.8.2.0")])
    assert "system Qt" in qd.format_report(env, qd.diagnose(env))


def test_import_failure_message_includes_the_original_error():
    message = qd.format_import_failure(
        ImportError("DLL load failed while importing QtCore: "
                    "The specified procedure could not be found.")
    )
    assert "DLL load failed" in message
    assert "sreeni-cli doctor" in message
    message.encode("ascii")


def test_import_failure_leads_with_the_diagnosis_when_there_is_one(monkeypatch):
    """This is the literal text a #92 user sees when the GUI dies, and on a healthy
    machine the probe finds nothing, so it needs a broken environment injected."""
    monkeypatch.setattr(qd, "qt_environment", _broken_env)
    message = qd.format_import_failure(ImportError("DLL load failed"))

    assert "Likely cause" in message
    assert qd.SOURCE_CONDA_LIBRARY_BIN in message
    assert "conda remove" in message
    message.encode("ascii")


def test_import_failure_never_raises_even_if_the_probe_breaks(monkeypatch):
    """This runs inside an exception handler. A traceback out of the diagnostic would
    bury the very failure it exists to explain."""
    monkeypatch.setattr(
        qd, "qt_environment", lambda: (_ for _ in ()).throw(RuntimeError("probe blew up"))
    )
    message = qd.format_import_failure(ImportError("boom"))
    assert "boom" in message
    assert "diagnostic itself failed" in message


# --- the live probe --------------------------------------------------------


def test_qt_environment_runs_on_this_machine():
    env = qd.qt_environment()
    for key in ("platform", "distributions", "qt_core_candidates", "msvc_candidates",
                "wheel_qt_bin", "wheel_qt_version", "package_dir"):
        assert key in env
    assert isinstance(qd.diagnose(env), list)


def test_this_machine_gets_a_clean_bill_of_health():
    """The suite is running under pytest-qt, so PyQt6 demonstrably imports here. A
    finding that would fail `doctor` is therefore a false positive by construction.

    This is the assertion that catches the whole class of bug the hand-built
    environments cannot: every one of those fixes `platform` to win32 and supplies
    candidates by hand, so a rule that misjudges a real Linux or macOS install looks
    perfectly correct to all of them. It runs on every CI leg.
    """
    findings = qd.diagnose(qd.qt_environment())
    failing = [f for f in findings if f.severity in qd.FAILING_SEVERITIES]
    assert not failing, (
        "doctor would fail on a machine where PyQt6 imports fine: "
        + "; ".join(f"[{f.severity}] {f.title} -- {f.detail}" for f in failing)
    )


def test_the_import_failure_path_may_be_noisier_but_must_not_crash():
    """`qt_failed=True` deliberately admits weaker evidence, so findings here are fine
    on a healthy machine -- but the rules still have to run cleanly against a real
    environment on every platform."""
    assert isinstance(qd.diagnose(qd.qt_environment(), qt_failed=True), list)


def test_the_probe_does_not_import_pyqt6():
    """`find_spec` locates PyQt6 without executing it. If this module ever imported
    PyQt6 for real, it would crash in exactly the environment it exists to diagnose."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from digitalsreeni_image_annotator.core import qt_diagnostics as qd;"
        "qd.diagnose(qd.qt_environment());"
        "print('loaded' if 'PyQt6' in sys.modules else 'clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="PE version resources are Windows-only")
def test_dll_file_version_reads_a_real_system_dll():
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                        "System32", "kernel32.dll")
    version = qd.dll_file_version(path)
    assert version is not None
    assert qd._version_tuple(version) is not None


@pytest.mark.skipif(sys.platform != "win32", reason="PE version resources are Windows-only")
def test_dll_file_version_returns_none_for_a_file_that_is_not_a_dll(tmp_path):
    """The probe runs over whatever is sitting in these directories, so a text file
    named Qt6Core.dll must produce 'unreadable', not an exception."""
    bogus = tmp_path / qd.QT_CORE_DLL
    bogus.write_bytes(b"not a real PE")
    assert qd.dll_file_version(str(bogus)) is None


def test_dll_file_version_returns_none_for_a_missing_file():
    assert qd.dll_file_version(os.path.join("C:\\", "nope", "absent.dll")) is None
