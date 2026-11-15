"""Stage 1: load the raw dataset and keep a validated copy in data/interim."""

import pandas as pd

from src.utils import data_dir, get_logger, project_path


def data_ingestion(config):
    """Load the raw CSV and save it to data/interim/ingested.csv.

    Args:
        config: Project configuration dictionary.

    Returns:
        The loaded DataFrame.
    """
    logger = get_logger(config)
    raw_path = project_path(config["data"]["path"])
    target = config["data"]["target"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"Raw data is empty: {raw_path}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {raw_path}")

    output_path = data_dir(config, "interim") / "ingested.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Ingested %s rows and %s columns", len(df), len(df.columns))
    return df
