"""Shared test fixtures: a small dataset and a matching configuration."""

import pandas as pd
import pytest

TARGET = "strength"


def build_config(tmp_path):
    """Return a config that reads and writes everything under tmp_path."""
    return {
        "data": {
            "path": str(tmp_path / "data" / "raw" / "concrete.csv"),
            "target": TARGET,
            "test_size": 0.5,
            "random_state": 1,
        },
        "preprocessing": {
            "handle_missing_values": True,
            "imputation_strategy": "median",
            "remove_duplicates": True,
        },
        "model": {
            "type": "decision_tree",
            "params": {"decision_tree": {"max_depth": 3}},
        },
        "artifacts": {
            "model_dir": str(tmp_path / "artifacts" / "models"),
            "report_dir": str(tmp_path / "artifacts" / "reports"),
        },
        "logging": {
            "level": "INFO",
            "file": str(tmp_path / "logs" / "test.log"),
        },
        "evaluation": {"metrics": ["r2", "mae", "rmse"]},
    }


@pytest.fixture
def dataset(tmp_path):
    """Write a small raw CSV and return its path."""
    raw_path = tmp_path / "data" / "raw" / "concrete.csv"
    raw_path.parent.mkdir(parents=True)

    pd.DataFrame(
        {
            "cement": [100, 200, 300, 400, 500, 600],
            "water": [10, 20, 30, 40, 50, 60],
            TARGET: [20, 30, 40, 50, 60, 70],
        }
    ).to_csv(raw_path, index=False)

    return raw_path


@pytest.fixture
def config(tmp_path, dataset):
    """Configuration pointing at the temporary dataset."""
    return build_config(tmp_path)
