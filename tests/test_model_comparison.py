"""Tests for the classroom model comparison."""
from __future__ import annotations

import pandas as pd

from src.model_comparison import run


def test_reports_are_created_and_metrics_are_valid(tmp_path) -> None:
    summary = run(output_dir=tmp_path, seed=42)
    assert len(summary) == 2
    assert set(summary["model"]) == {"Logistic Regression", "k-Nearest Neighbors"}
    assert summary["holdout_accuracy"].between(0, 1).all()
    assert summary["holdout_macro_f1"].between(0, 1).all()
    assert summary["cv_accuracy_mean"].between(0, 1).all()
    assert (tmp_path / "model_comparison.csv").exists()
    assert len(list(tmp_path.glob("confusion_*.csv"))) == 2


def test_results_are_reproducible(tmp_path) -> None:
    x = run(output_dir=tmp_path / "first", seed=17)
    y = run(output_dir=tmp_path / "second", seed=17)
    pd.testing.assert_frame_equal(x, y)
