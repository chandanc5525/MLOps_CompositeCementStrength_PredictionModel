"""Stage 4: score the saved model on the test data."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import (
    data_dir,
    get_logger,
    load_json,
    load_model,
    metadata_path,
    model_path,
    report_path,
    save_json,
)

# Metric names from config.yaml -> scoring functions.
METRICS = {
    "r2": r2_score,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
}


def calculate_metrics(y_true, y_pred, metric_names):
    """Compute the requested metrics and return them as a dictionary.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        metric_names: Metric names listed under evaluation.metrics in config.

    Returns:
        A dictionary mapping each metric name to its value.
    """
    results = {}

    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Choose from {list(METRICS)}")
        results[name] = float(METRICS[name](y_true, y_pred))

    return results


def evaluate_model(config):
    """Evaluate the saved model on test.csv and write a JSON report.

    Args:
        config: Project configuration dictionary.

    Returns:
        A dictionary with the metric values.
    """
    logger = get_logger(config)
    test_path = data_dir(config, "processed") / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. Run the preprocess stage first."
        )
    if not model_path(config).exists():
        raise FileNotFoundError("No trained model found. Run the train stage first.")

    model = load_model(model_path(config))
    metadata = load_json(metadata_path(config))

    df = pd.read_csv(test_path)
    target = config["data"]["target"]
    predictions = model.predict(df[metadata["features"]])
    metrics = calculate_metrics(
        df[target], predictions, config["evaluation"]["metrics"]
    )

    report = {
        "model_type": metadata["model_type"],
        "test_rows": len(df),
        "metrics": metrics,
    }
    save_json(report, report_path(config, "evaluation.json"))

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
