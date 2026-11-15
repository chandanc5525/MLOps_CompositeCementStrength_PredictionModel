"""Stage 5: predict cement strength with the saved model."""

import pandas as pd

from src.utils import get_logger, load_json, load_model, metadata_path, model_path


def predict(config, input_path, output_path=None):
    """Predict on a CSV file and optionally save the result.

    Args:
        config: Project configuration dictionary.
        input_path: CSV containing the same feature columns used for training.
        output_path: Optional path to save the CSV with a prediction column.

    Returns:
        The input DataFrame with an added 'prediction' column.
    """
    logger = get_logger(config)

    if not model_path(config).exists():
        raise FileNotFoundError("No trained model found. Run the train stage first.")

    model = load_model(model_path(config))
    metadata = load_json(metadata_path(config))
    features = metadata["features"]

    df = pd.read_csv(input_path)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df["prediction"] = model.predict(df[features])

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("Saved %s predictions to %s", len(df), output_path)

    return df
