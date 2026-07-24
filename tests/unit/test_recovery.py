"""Unit tests for the unsaved-project recovery store (#41).

The recovery *directory* is redirected into tmp_path (monkeypatching
``recovery_dir``) and the QSettings pointer is an INI file, so nothing touches
the real per-user AppData / registry.
"""

import json
import os

import pytest
from PyQt6.QtCore import QSettings

from digitalsreeni_image_annotator.core import recovery


@pytest.fixture
def rec_dir(tmp_path, monkeypatch):
    d = tmp_path / "recovery"
    d.mkdir()
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(d))
    return d


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


SAMPLE = {
    "classes": [{"name": "cell", "color": "#ff0000"}],
    "images": [],
    "notes": "hi",
}


def test_write_then_pending_returns_path(rec_dir, settings):
    path = recovery.write_recovery(SAMPLE, settings)
    assert os.path.exists(path)
    assert recovery.pending_recovery(settings) == path


def test_written_content_roundtrips(rec_dir, settings):
    path = recovery.write_recovery(SAMPLE, settings)
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == SAMPLE


def test_pending_with_deleted_file_returns_none_and_clears_key(rec_dir, settings):
    path = recovery.write_recovery(SAMPLE, settings)
    os.remove(path)
    assert recovery.pending_recovery(settings) is None
    # The stale pointer is cleared, so a second probe is also None.
    assert settings.value("recovery/pending_path", "", type=str) == ""


def test_clear_removes_file_and_key(rec_dir, settings):
    path = recovery.write_recovery(SAMPLE, settings)
    recovery.clear_recovery(settings)
    assert not os.path.exists(path)
    assert recovery.pending_recovery(settings) is None


def test_clear_is_idempotent_when_nothing_pending(rec_dir, settings):
    recovery.clear_recovery(settings)
    recovery.clear_recovery(settings)  # must not raise


def test_write_leaves_no_tmp_file(rec_dir, settings):
    recovery.write_recovery(SAMPLE, settings)
    assert not any(p.name.endswith(".tmp") for p in rec_dir.iterdir())


def test_second_write_overwrites_single_file(rec_dir, settings):
    recovery.write_recovery(SAMPLE, settings)
    recovery.write_recovery({**SAMPLE, "notes": "second"}, settings)
    files = list(rec_dir.iterdir())
    assert len(files) == 1
    with open(files[0], encoding="utf-8") as f:
        assert json.load(f)["notes"] == "second"
