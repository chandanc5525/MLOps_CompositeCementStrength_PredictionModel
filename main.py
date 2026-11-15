"""Command line entry point for the cement strength pipeline.

Each stage can be run on its own, or all stages at once with run-all:

    python main.py ingest
    python main.py preprocess
    python main.py train
    python main.py evaluate
    python main.py predict --input data/raw/Concrete_Data.csv --output predictions.csv
    python main.py run-all
    python main.py serve
"""

import argparse

import uvicorn

from src.api import create_app
from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data
from src.model_building import train_model
from src.model_evaluation import evaluate_model
from src.model_prediction import predict
from src.utils import load_config


def run_all(config):
    """Run ingest, preprocess, train and evaluate in order."""
    data_ingestion(config)
    preprocess_data(config)
    train_model(config)
    evaluate_model(config)


def main():
    """Parse command line arguments and run the requested stage."""
    parser = argparse.ArgumentParser(
        description="Composite Cement Strength Prediction pipeline"
    )
    parser.add_argument(
        "-c", "--config", default=None, help="Path to config.yaml"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("ingest", help="Load and validate the raw dataset")
    commands.add_parser("preprocess", help="Clean and split the dataset")
    commands.add_parser("train", help="Train and save the model")
    commands.add_parser("evaluate", help="Evaluate the saved model")
    commands.add_parser("run-all", help="Run ingest, preprocess, train and evaluate")

    predict_parser = commands.add_parser("predict", help="Predict on a CSV file")
    predict_parser.add_argument("--input", required=True, help="Input CSV path")
    predict_parser.add_argument("--output", default=None, help="Output CSV path")

    serve_parser = commands.add_parser("serve", help="Start the web console and API")
    serve_parser.add_argument("--host", default=None, help="Server host")
    serve_parser.add_argument("--port", type=int, default=None, help="Server port")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "ingest":
        data_ingestion(config)
    elif args.command == "preprocess":
        preprocess_data(config)
    elif args.command == "train":
        train_model(config)
    elif args.command == "evaluate":
        evaluate_model(config)
    elif args.command == "predict":
        result = predict(config, args.input, args.output)
        if not args.output:
            print(result.to_string())
    elif args.command == "run-all":
        run_all(config)
    elif args.command == "serve":
        serve_config = config.get("serve", {})
        uvicorn.run(
            create_app(args.config),
            host=args.host or serve_config.get("host", "127.0.0.1"),
            port=args.port or serve_config.get("port", 8000),
        )


if __name__ == "__main__":
    main()
