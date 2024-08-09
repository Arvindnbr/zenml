"Registers model in MLFlow"
import os
from pathlib import Path
import csv
import logging
import yaml
import cloudpickle
import pandas as pd
import mlflow
from zenml import step
from zenml.logger import get_logger
from scripts.utils import wrapper
from scripts.utils.wrapper import YoloWrapper
from typing import Annotated, Any, Dict, Tuple


logger = get_logger(__name__)


def get_experiment_id(name: str):
    """Retrieve experiment if registered name, else create experiment.

    Args:
        name (str): Mlflow experiment name

    Returns:
        str: Mlfow experiment id
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def read_lines(path: str):
    """Given a path to a file, this function reads file, seperates the lines and returns a list of those separated lines.

    Args:
        path (str): Path to file

    Returns:
        list: List made of of file lines
    """
    with open(path) as f:
        return f.read().splitlines()


def log_metrics(save_dir: str, log_results: bool = True):
    """Log metrics to Mlflow from the Yolo model outputs.

    Args:
        save_dir (str): Path to Yolo save directory, i.e - runs/train
        log_results (bool): If True, the results are logged to MLflow server
    """
    save_dir = Path(save_dir)
    try:
        with open(save_dir / "results.csv", "r") as csv_file:
            metrics_reader = csv.DictReader(csv_file)
            metrics_list = []
            for metrics in metrics_reader:
                # Create an empty dictionary to store the updated key-value pairs for this row
                updated_metrics = {}
                # Iterate through the key-value pairs in this row's dictionary
                for key, value in metrics.items():
                    # Remove whitespace from the key
                    key = key.strip()
                    value = value.strip()
                    # Remove extra strings in keys
                    patterns = ["(B)", "metrics/"]
                    for pattern in patterns:
                        key = key.replace(pattern, "")
                    # Add the updated key-value pair to the updated row dictionary
                    try:
                        # Add the updated key-value pair to the updated row dictionary
                        updated_metrics[key] = float(value)
                    except ValueError:
                        logging.error(f"ValueError: Could not convert {value} to float.")
                    metrics_list.append(updated_metrics)
                    if log_results:
                        mlflow.log_metrics(updated_metrics)
        return metrics_list
    except FileNotFoundError:
        print(f"FileNotFoundError: Could not find {save_dir / 'results.csv'}.")
    except IOError:
        print(f"IOError: Could not read {save_dir / 'results.csv'}.")



@step
def register_model(experiment_name: str, model_name: str, save_dir: str
                   ) -> Tuple[Annotated[str, "Run_id"],
                              Annotated[str, "Experiment_id"]]:
    
    """
    Registers a model with mlflow

    Args:
        experiment_name (str): Name of Mlfow experiment
        model_name (str): Name that will be registered with Mlflow
        save_dir (Path): Path object where the results of the Yolo model are saved. I.e 'runs' directory
    """
    save_dir = Path(save_dir)
    logging.debug(f"Save Directory: {save_dir}")

    model_path = f"{save_dir}/weights/best.pt"
    artifacts = {"path": model_path}

    model = YoloWrapper()

    exp_id = get_experiment_id(experiment_name)
    
    #cloudpickle.register_pickle_by_value(wrapper)

    with mlflow.start_run(experiment_id=exp_id) as run:
        # Log some params
        with open(save_dir / "args.yaml", "r") as param_file:
            params = yaml.safe_load(param_file)
        mlflow.log_params(params)

        log_metrics(save_dir, True)
        mlflow.log_artifact(f"{save_dir}/weights/best.pt")
        pip_reqs = read_lines("requirements.txt")
        mlflow.pyfunc.log_model(
            "model",
            python_model=model,
            pip_requirements=pip_reqs,
            artifacts=artifacts,
            registered_model_name=model_name,
        )
        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        logging.info(f"artifact_uri = {mlflow.get_artifact_uri()}")
        logging.info(f"runID: {run_id}")
        logging.info(f"experiment_id: {experiment_id}")
        return run_id, experiment_id


