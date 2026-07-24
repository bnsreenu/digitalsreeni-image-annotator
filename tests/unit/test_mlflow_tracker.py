"""Unit tests for the always-on MLflow tracker (issue #74).

Tracking is mandatory (MLflow is a core dependency), so there is no "disabled"
mode. These tests pin two contracts: (1) the ``_NullTracker`` no-op used when a
trainer is invoked without a tracker is inert, and (2) crash-safety — if MLflow
is unexpectedly broken at run time, ``start()`` degrades to untracked without
raising, so a training job is never killed by a tracking failure. The
live-logging path runs only when mlflow is importable (skipped otherwise).
"""

import os
import sys

import pytest

from digitalsreeni_image_annotator.training import mlflow_tracker
from digitalsreeni_image_annotator.training.mlflow_tracker import (
    MLflowTracker,
    _NullTracker,
    resolve_tracking_uri,
)


class TestNullTracker:
    def test_all_methods_are_noops_and_inactive(self):
        # Stand-in used when a trainer gets no tracker (direct calls, tests).
        t = _NullTracker()
        t.set_log(lambda m: None)
        t.set_run_url_callback(lambda url: None)
        assert t.start({"a": 1}) is False
        assert t.active is False
        # None of these raise, even though no run is open.
        t.log_metrics({"loss": 0.5}, step=1)
        t.log_artifact("/nonexistent/path")
        t.end()


class TestRunUiUrl:
    def test_url_format(self):
        assert mlflow_tracker.run_ui_url("123", "abc", port=5000) == (
            "http://localhost:5000/#/experiments/123/runs/abc"
        )


class TestCrashSafety:
    def test_broken_mlflow_degrades_without_raising(self, monkeypatch):
        # Simulate mlflow being unimportable at start() time: a None entry in
        # sys.modules makes `import mlflow` raise ImportError.
        monkeypatch.setitem(sys.modules, "mlflow", None)
        msgs = []
        t = MLflowTracker(tracking_uri="x", log=msgs.append)
        assert t.start({"a": 1}) is False   # never raises
        assert t.active is False
        t.log_metrics({"loss": 1.0})        # safe no-op
        t.log_artifact("/nope")
        t.end()
        assert any("unavailable" in m for m in msgs)


class TestMlflowAvailable:
    def test_probe_is_cached(self, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        first = mlflow_tracker.mlflow_available()
        # Second call must return the cached value without re-probing.
        assert mlflow_tracker.mlflow_available() is first


class _FakeMW:
    def __init__(self, project_dir=None):
        if project_dir is not None:
            self.current_project_dir = project_dir


class TestResolveTrackingUri:
    def test_override_wins(self, monkeypatch, tmp_path):
        # load_mlflow_prefs is imported inside the function, so patch the source.
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: (str(tmp_path / "override"), "exp"),
        )
        assert resolve_tracking_uri(_FakeMW(project_dir="/proj")) == str(
            tmp_path / "override"
        )

    def test_project_dir_used_when_no_override(self, monkeypatch):
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: ("", "exp"),
        )
        uri = resolve_tracking_uri(_FakeMW(project_dir="/proj"))
        assert uri == os.path.join("/proj", "mlruns")

    def test_cwd_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: ("", "exp"),
        )
        uri = resolve_tracking_uri(_FakeMW(project_dir=None))
        assert uri == os.path.join(os.getcwd(), "mlruns")


class TestLiveLogging:
    """Exercised only when mlflow is actually installed."""

    def test_run_records_params_and_metrics(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        if not mlflow_tracker.mlflow_available():
            pytest.skip("mlflow not installed")
        import mlflow

        store = str(tmp_path / "mlruns")
        t = MLflowTracker(
            tracking_uri=store, experiment_name="ut-exp",
            run_name="ut-run",
        )
        assert t.start({"epochs": 3, "lr": 1e-4, "skip": None}) is True
        t.log_metrics({"loss": 0.9}, step=1)
        t.log_metrics({"loss": 0.4}, step=2)
        t.end()
        assert t.active is False

        mlflow.set_tracking_uri(mlflow_tracker.to_mlflow_uri(store))
        exp = mlflow.get_experiment_by_name("ut-exp")
        assert exp is not None
        runs = mlflow.search_runs([exp.experiment_id])
        assert len(runs) == 1
        assert runs.iloc[0]["params.epochs"] == "3"
        # None-valued params are filtered out.
        assert "params.skip" not in runs.columns or runs.iloc[0].get(
            "params.skip"
        ) is None
        assert runs.iloc[0]["metrics.loss"] == 0.4

    def test_artifact_is_logged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        if not mlflow_tracker.mlflow_available():
            pytest.skip("mlflow not installed")
        import mlflow

        # Stand-in for a saved checkpoint.
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake-weights")

        store = str(tmp_path / "mlruns")
        t = MLflowTracker(
            tracking_uri=store, experiment_name="ut-art",
            run_name="ut-run",
        )
        assert t.start({"epochs": 1}) is True
        run_id = mlflow.active_run().info.run_id
        t.log_artifact(str(ckpt))
        # A non-existent path is a safe no-op (must not raise or abort).
        t.log_artifact(str(tmp_path / "missing.pt"))
        t.end()

        mlflow.set_tracking_uri(mlflow_tracker.to_mlflow_uri(store))
        artifacts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
        assert "model.pt" in artifacts

    def test_double_start_self_heals_stranded_run(self, tmp_path, monkeypatch):
        """A run left active (killed before end()) must not poison the next
        start() — the tracker closes the stranded run instead of degrading to
        untracked. See start()'s self-heal branch."""
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        if not mlflow_tracker.mlflow_available():
            pytest.skip("mlflow not installed")

        store = str(tmp_path / "mlruns")
        t = MLflowTracker(
            tracking_uri=store, experiment_name="ut-heal",
            run_name="ut-run",
        )
        assert t.start({"epochs": 1}) is True   # opens run A, never ended
        # Second start() with a run still active must still come up live.
        assert t.start({"epochs": 2}) is True
        assert t.active is True
        t.end()

    def test_run_url_callback_fires_with_deep_link(self, tmp_path, monkeypatch):
        """On start(), the tracker captures run/experiment ids and hands the
        GUI a deep link to that run (clickable link + auto-open)."""
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        if not mlflow_tracker.mlflow_available():
            pytest.skip("mlflow not installed")

        store = str(tmp_path / "mlruns")
        seen = []
        t = MLflowTracker(
            tracking_uri=store, experiment_name="ut-url", run_name="ut-run",
        )
        t.set_run_url_callback(seen.append)
        assert t.start({"epochs": 1}) is True
        t.end()

        assert len(seen) == 1
        assert t.run_id and t.experiment_id
        assert seen[0] == mlflow_tracker.run_ui_url(t.experiment_id, t.run_id)
        assert t.run_id in seen[0] and t.experiment_id in seen[0]


class TestEmitCrashSafety:
    def test_emit_swallows_and_logs_broken_sink(self):
        """A raising log sink must not propagate out of ``_emit``; the dropped
        message is logged via the module logger instead (issue #34). Capture is
        done by attaching a handler to ``mlflow_tracker.logger`` directly, which
        is robust to the package logger's ``propagate = False`` (ADR-030)."""
        import logging

        def broken_sink(msg):
            raise RuntimeError("sink is broken")

        t = MLflowTracker(tracking_uri="x", log=broken_sink)

        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())
        old_level = mlflow_tracker.logger.level
        mlflow_tracker.logger.addHandler(handler)
        mlflow_tracker.logger.setLevel(logging.DEBUG)
        try:
            t._emit("hello")  # must NOT raise even though the sink throws
        finally:
            mlflow_tracker.logger.removeHandler(handler)
            mlflow_tracker.logger.setLevel(old_level)

        assert any("hello" in m for m in records)
