"""Stage 3: build and train the regression model described in config.yaml."""

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import (
    data_dir,
    get_logger,
    metadata_path,
    model_path,
    save_json,
    save_model,
)

# Model types supported by config.yaml -> scikit-learn estimators.
MODELS = {
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
}


def build_model(config):
    """Create an untrained model from the settings in config.yaml.

    Args:
        config: Project configuration dictionary.

    Returns:
        A scikit-learn estimator.
    """
    model_type = config["model"]["type"]

    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from {list(MODELS)}"
        )

    params = config["model"]["params"].get(model_type, {})
    return MODELS[model_type](**params)


def train_model(config):
    """Train the configured model on train.csv and save model plus metadata.

    Args:
        config: Project configuration dictionary.

    Returns:
        The trained model.
    """
    logger = get_logger(config)
    train_path = data_dir(config, "processed") / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Run the preprocess stage first."
        )

    df = pd.read_csv(train_path)
    target = config["data"]["target"]
    features = df.drop(columns=[target])
    labels = df[target]

    model = build_model(config)
    model.fit(features, labels)

    save_model(model, model_path(config))
    metadata = {
        "model_type": config["model"]["type"],
        "params": config["model"]["params"].get(config["model"]["type"], {}),
        "target": target,
        "features": list(features.columns),
        "training_rows": len(df),
    }
    save_json(metadata, metadata_path(config))

    logger.info(
        "Trained %s on %s rows", config["model"]["type"], len(df)
    )
    return model
