"""End-to-end test of the pipeline using a small temporary dataset."""

from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data
from src.model_building import train_model
from src.model_evaluation import evaluate_model
from src.model_prediction import predict


def test_pipeline_runs_end_to_end(config, dataset, tmp_path):
    """Every stage runs and produces its output files."""
    data_ingestion(config)
    preprocess_data(config)
    train_model(config)
    metrics = evaluate_model(config)

    assert set(metrics) == {"r2", "mae", "rmse"}
    assert (tmp_path / "artifacts" / "models" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "reports" / "evaluation.json").exists()

    result = predict(config, str(dataset))

    assert len(result) == 6
    assert "prediction" in result.columns
