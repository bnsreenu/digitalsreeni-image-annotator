"""Unit tests for app_settings (UI preference persistence).

QSettings is exercised against an INI file in tmp_path so the tests
never touch the real per-user registry/config.
"""

import pytest
from PyQt6.QtCore import QSettings

from digitalsreeni_image_annotator.app_settings import (
    FONT_PT_DEFAULT,
    FONT_PT_MAX,
    FONT_PT_MIN,
    MLFLOW_EXPERIMENT_DEFAULT,
    clamp_font_pt,
    load_mlflow_prefs,
    load_ui_prefs,
    save_mlflow_prefs,
    save_ui_prefs,
)


class TestClampFontPt:
    def test_in_range_passes_through(self):
        assert clamp_font_pt(12) == 12

    def test_below_min_clamps(self):
        assert clamp_font_pt(3) == FONT_PT_MIN

    def test_above_max_clamps(self):
        assert clamp_font_pt(99) == FONT_PT_MAX

    def test_numeric_string_is_coerced(self):
        # QSettings INI backend round-trips ints as strings.
        assert clamp_font_pt("14") == 14

    def test_garbage_falls_back_to_default(self):
        assert clamp_font_pt("huge") == FONT_PT_DEFAULT

    def test_none_falls_back_to_default(self):
        assert clamp_font_pt(None) == FONT_PT_DEFAULT


@pytest.fixture
def ini_settings(tmp_path):
    return QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)


class TestUiPrefsRoundtrip:
    def test_defaults_from_empty_settings(self, ini_settings):
        assert load_ui_prefs(ini_settings) == (FONT_PT_DEFAULT, True)

    def test_roundtrip(self, ini_settings):
        save_ui_prefs(18, False, ini_settings)
        ini_settings.sync()
        assert load_ui_prefs(ini_settings) == (18, False)

    def test_save_clamps_out_of_range(self, ini_settings):
        save_ui_prefs(100, True, ini_settings)
        assert load_ui_prefs(ini_settings) == (FONT_PT_MAX, True)

    def test_load_clamps_corrupt_value(self, ini_settings):
        ini_settings.setValue("ui/font_pt", "not-a-number")
        assert load_ui_prefs(ini_settings)[0] == FONT_PT_DEFAULT


class TestMlflowPrefsRoundtrip:
    # Tracking is always on; only the destination (URI/experiment) is stored.
    def test_defaults_from_empty_settings(self, ini_settings):
        assert load_mlflow_prefs(ini_settings) == ("", MLFLOW_EXPERIMENT_DEFAULT)

    def test_roundtrip(self, ini_settings):
        save_mlflow_prefs("/tmp/mlruns", "my-exp", ini_settings)
        ini_settings.sync()
        assert load_mlflow_prefs(ini_settings) == ("/tmp/mlruns", "my-exp")

    def test_blank_experiment_falls_back_to_default(self, ini_settings):
        save_mlflow_prefs("", "   ", ini_settings)
        assert load_mlflow_prefs(ini_settings) == ("", MLFLOW_EXPERIMENT_DEFAULT)

    def test_uri_is_stripped(self, ini_settings):
        save_mlflow_prefs("  /data/runs  ", "exp", ini_settings)
        assert load_mlflow_prefs(ini_settings)[0] == "/data/runs"
