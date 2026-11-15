"""FastAPI app that serves the web console and prediction endpoints."""

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils import (
    data_dir,
    load_config,
    load_json,
    load_model,
    metadata_path,
    model_path,
    project_path,
    report_path,
)


class PredictRequest(BaseModel):
    """Rows to score; each row maps a feature name to its value."""

    rows: list[dict[str, float]]


class PredictResponse(BaseModel):
    """Predicted strength values, in the same order as the request rows."""

    predictions: list[float]


def create_app(config=None):
    """Build the FastAPI app with all endpoints and the frontend.

    Args:
        config: Configuration dictionary, or a path to a config file.
            Defaults to config.yaml in the project root.

    Returns:
        A FastAPI application instance.
    """
    if not isinstance(config, dict):
        config = load_config(config)

    app = FastAPI(
        title="Composite Cement Strength Prediction",
        description="Predict concrete compressive strength (MPa).",
    )
    cache = {"model": None, "metadata": None}

    def model_and_metadata():
        """Load the model and metadata once, then reuse them for later requests."""
        if cache["model"] is None:
            if not model_path(config).exists():
                raise HTTPException(
                    status_code=404,
                    detail="No trained model found. Run: python main.py train",
                )
            cache["model"] = load_model(model_path(config))
            cache["metadata"] = load_json(metadata_path(config))
        return cache["model"], cache["metadata"]

    @app.get("/api/health")
    def health():
        """Report whether the service is up and a model is available."""
        return {
            "status": "ok",
            "model_available": model_path(config).exists(),
        }

    @app.get("/api/model")
    def model_info():
        """Return the model type, target and latest evaluation metrics."""
        _, metadata = model_and_metadata()
        metrics = None
        if report_path(config, "evaluation.json").exists():
            metrics = load_json(report_path(config, "evaluation.json"))["metrics"]

        return {
            "model_type": metadata["model_type"],
            "target": metadata["target"],
            "features": metadata["features"],
            "metrics": metrics,
        }

    @app.get("/api/features")
    def features():
        """Return the feature names the model expects."""
        _, metadata = model_and_metadata()
        return {"features": metadata["features"]}

    @app.get("/api/sample")
    def sample():
        """Return one example row from the test data."""
        test_path = data_dir(config, "processed") / "test.csv"
        if not test_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Test data not found. Run: python main.py preprocess",
            )

        _, metadata = model_and_metadata()
        row = pd.read_csv(test_path).iloc[0]
        return {"sample": {name: float(row[name]) for name in metadata["features"]}}

    @app.post("/api/predict", response_model=PredictResponse)
    def make_predictions(request: PredictRequest):
        """Predict the strength for each row in the request."""
        if not request.rows:
            raise HTTPException(status_code=400, detail="rows must not be empty")

        model, metadata = model_and_metadata()
        expected = metadata["features"]

        for index, row in enumerate(request.rows):
            missing = [name for name in expected if name not in row]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {index} is missing features: {missing}",
                )

        frame = pd.DataFrame(request.rows)[expected]
        predictions = model.predict(frame)
        return {"predictions": [float(value) for value in predictions]}

    frontend_dir = project_path("frontend")
    if frontend_dir.exists():
        app.mount(
            "/", StaticFiles(directory=frontend_dir, html=True), name="frontend"
        )

    return app
