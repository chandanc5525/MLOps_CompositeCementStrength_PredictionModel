"""Tests for the REST API and the served web console."""

from fastapi.testclient import TestClient

from src.api import create_app
from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data
from src.model_building import train_model
from src.model_evaluation import evaluate_model


def build_client(config):
    """Train a model on the temporary data and return a test client."""
    data_ingestion(config)
    preprocess_data(config)
    train_model(config)
    evaluate_model(config)
    return TestClient(create_app(config))


def test_frontend_and_model_info(config):
    """The web console is served and model details are available."""
    client = build_client(config)

    assert client.get("/").status_code == 200
    assert client.get("/js/app.js").status_code == 200
    assert client.get("/css/style.css").status_code == 200

    model = client.get("/api/model").json()
    assert model["model_type"] == "decision_tree"
    assert set(model["metrics"]) == {"r2", "mae", "rmse"}

    features = client.get("/api/features").json()["features"]
    assert features == ["cement", "water"]


def test_predict_single_row(config):
    """A valid row returns one prediction."""
    client = build_client(config)
    sample = client.get("/api/sample").json()["sample"]

    response = client.post("/api/predict", json={"rows": [sample]})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_validates_input(config):
    """Missing features are rejected and non-numeric values fail validation."""
    client = build_client(config)

    missing = client.post("/api/predict", json={"rows": [{"cement": 100}]})
    assert missing.status_code == 400

    invalid = client.post("/api/predict", json={"rows": [{"cement": "lots"}]})
    assert invalid.status_code == 422


def test_predict_without_model(config):
    """Predicting before training returns a clear 404."""
    client = TestClient(create_app(config))

    response = client.post(
        "/api/predict", json={"rows": [{"cement": 100, "water": 10}]}
    )

    assert response.status_code == 404
