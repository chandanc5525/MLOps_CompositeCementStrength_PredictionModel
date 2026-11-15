"""Shared helpers for the cement strength pipeline.

Everything that is used by more than one stage lives here: loading the
configuration, resolving relative paths, logging, JSON I/O and saving or
loading the trained model.
"""

import json
import logging
from pathlib import Path

import joblib
import yaml

# Project root is one level above the src/ folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(config_path=None):
    """Load the YAML configuration file.

    Args:
        config_path: Optional path to a config file. Defaults to config.yaml
            in the project root.

    Returns:
        The configuration as a dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path):
    """Return an absolute path, resolving relative paths from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def data_dir(config, stage):
    """Return a data sub-folder such as data/raw, data/interim or data/processed."""
    raw_path = project_path(config["data"]["path"])
    return raw_path.parent.parent / stage


def model_path(config):
    """Return the path of the saved model file."""
    return project_path(config["artifacts"]["model_dir"]) / "model.joblib"


def metadata_path(config):
    """Return the path of the model metadata file."""
    return project_path(config["artifacts"]["model_dir"]) / "model_metadata.json"


def report_path(config, name):
    """Return a report path inside the configured report folder."""
    return project_path(config["artifacts"]["report_dir"]) / name


def get_logger(config, name="pipeline"):
    """Create a logger that writes to the console and to the log file in config.

    The same logger is reused on later calls so messages are not duplicated.
    """
    log_file = project_path(config["logging"]["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def save_json(data, path):
    """Write a dictionary to a JSON file, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path):
    """Read a JSON file and return its content as a dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model, path):
    """Save a trained model with joblib, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a model file saved with joblib."""
    return joblib.load(path)
