from scripts.entity.exception import AppException
import sys, os, yaml
from mlflow import MlflowClient
from zenml.logger import get_logger
from zenml import pipeline
from ultralytics import YOLO
from zenml.client import Client
from scripts.steps.inference import yolov8_validation_step
from scripts.steps.train import load_model, Trainer
from scripts.steps.log_mlflow import register_model
from scripts.steps.best_model import production_model
from scripts.config.configuration import ConfigurationManager
from scripts.steps.retraining import Retrain_dataset, retrain
from typing import Annotated



logger = get_logger(__name__)

config = ConfigurationManager()
threshold = config.get_threshold()
eval = config.get_evaluation()
params = config.get_params()
train_config = config.get_train_log_config()
client = MlflowClient(tracking_uri=train_config.mlflow_uri)

@pipeline
def inference() -> Annotated[str,"dataset_root"]:

    model_versions = client.search_model_versions(f"name = '{eval.name}'")

    production_run_id = None
    for version in model_versions:
        if version.current_stage == "Production":
            production_run_id = version.run_id
            break

    run_info = client.get_run(production_run_id)
    artifact_uri = run_info.info.artifact_uri.replace("mlflow-artifacts:", "mlartifacts")
    model = f"{artifact_uri}/model/artifacts/best.pt"
    with open(f"{artifact_uri}/args.yaml", 'r') as f:
        args = yaml.safe_load(f)
    dataset_root = os.path.split(args.get('data'))[0]
    val_status, metric = yolov8_validation_step(model_path=model,
                                        threshold=threshold.mAP50,
                                        validation_name = eval.name
                                        )
    
    with open(os.path.join(dataset_root,"data_validation","trigger.txt"), 'r') as file:
        trigger = file.readline()
    
    if trigger == "False":
        logger.info("Model is healthy!! continue with prediction")
    elif trigger == "True":
        new_dataset = Retrain_dataset(dataset_root)
        yolo_model = load_model(train_config.model)
        model, results, names, save_dir = retrain(config=params,
                                                    dataset=new_dataset,
                                                    yolo_model=yolo_model
                                                    )                                            
        os.environ["MLFLOW_TRACKING_URI"] = train_config.mlflow_uri
        run_id, experiment_id = register_model(experiment_name = train_config.experiment_name,
                                                    model_name = train_config.model_name,
                                                    save_dir = save_dir
                                                    )
        name, version, model = production_model(run_id, threshold)

    
     




if __name__ == "__main__":
    try:
        inference()
    except Exception as e:
        raise AppException(e, sys)
