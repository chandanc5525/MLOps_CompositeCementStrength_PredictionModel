"""Stage 2: clean the data, fill missing values and split train/test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import data_dir, get_logger


def clean_data(df, config):
    """Return a cleaned copy of the data.

    Rows without a target value are dropped, duplicates are removed and
    missing values are filled with the configured imputation strategy.

    Args:
        df: Raw DataFrame.
        config: Project configuration dictionary.

    Returns:
        A cleaned DataFrame.
    """
    target = config["data"]["target"]
    settings = config.get("preprocessing", {})

    df = df.dropna(subset=[target]).copy()

    if settings.get("remove_duplicates", True):
        df = df.drop_duplicates()

    if settings.get("handle_missing_values", True):
        strategy = settings.get("imputation_strategy", "median")
        if strategy == "median":
            fill_values = df.median(numeric_only=True)
        elif strategy == "mean":
            fill_values = df.mean(numeric_only=True)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        df = df.fillna(fill_values)

    # Drop any remaining rows that still contain missing values.
    return df.dropna()


def preprocess_data(config):
    """Clean the ingested data and save train.csv and test.csv.

    Args:
        config: Project configuration dictionary.

    Returns:
        A tuple with the training and test DataFrames.
    """
    logger = get_logger(config)
    input_path = data_dir(config, "interim") / "ingested.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Interim data not found: {input_path}. Run the ingest stage first."
        )

    df = clean_data(pd.read_csv(input_path), config)

    target = config["data"]["target"]
    features = df.drop(columns=[target])
    labels = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    output_dir = data_dir(config, "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info("Prepared %s train rows and %s test rows", len(train_df), len(test_df))
    return train_df, test_df
