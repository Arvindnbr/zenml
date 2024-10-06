import os, mlflow
from PIL import Image
from mlflow import MlflowClient
from zenml.logger import get_logger
from typing import Annotated, Any, Dict, Tuple
from zenml import step, log_artifact_metadata, ArtifactConfig
from ultralytics import YOLO
from zenml.metadata.metadata_types import DType
from scripts.config.configuration import ConfigurationManager

logger = get_logger(__name__)

config = ConfigurationManager()
train_config = config.get_train_log_config()
client = MlflowClient(tracking_uri=train_config.mlflow_uri)

@step(enable_cache=False)
def yolov8_prediction(model_path: str,
                      image_source: str,
                      save_path: str,
                      model_name : str
                      ):
    """Performs prediction using a YOLOv8 model and saves the results.

    Args:
        model_path (str): Path to the YOLO model.
        image_source (str): Path to the image or folder of images.
        save_path (str): path_to save the results.
    """
    predictions = []
    model = YOLO(model_path)
    results = model.predict(source=image_source, save=True, project=save_path, name=model_name, save_txt = True)

    for r in results:
        prediction_details = {
            "image_path": r.path,
            "boxes": r.boxes.xyxy.cpu().numpy().tolist(),  # bounding boxes
            "confidences": r.boxes.conf.cpu().numpy().tolist(),  # confidence scores
            "classes": r.boxes.cls.cpu().numpy().tolist(),  # class indices
            }
        predictions.append(prediction_details)
        
    return predictions
            



@step(enable_cache=False)
def yolov8_validation_step(model_path: str,
                           threshold: float,
                           validation_name:str) -> Tuple[Annotated[str, "Retrain_trigger"],
                           Annotated[Dict[str, Any], ArtifactConfig(name="infered_metrics",is_model_artifact=True)]]:
    """Validates the YOLOv8 model and checks if retraining is needed.

    Args:
        model_path (str): Path to the YOLO model.
        dataset_config (str): Path to the dataset configuration file.
        threshold (float): Threshold value for performance metrics.
        validation_project (str): Path to the project folder where validation results will be saved.
        validation_name (str): Name of the subfolder to save validation results.

    Returns:
        bool: True if retraining is needed, False otherwise.
        metric: the validation results.
        dataset_root: root dataset path.
    """

    model = YOLO(model_path)
    project_root = model.ckpt["train_args"]["project"]
    validation_root = f"{project_root}/validation"
    dataset_config = model.ckpt["train_args"]["data"]
    metrics = model.val(data=dataset_config, split="test", project=validation_root, name=validation_name)
    print(f"mAP50 of model : {metrics.box.map50}")
    print(f"mAP50-95 of model : {metrics.box.map}")

    experiment = client.get_experiment_by_name("demov8")

    log_artifact_metadata(artifact_name="infered_metrics",
                          metadata={
                              "mAP50": DType(metrics.box.map50),
                              "mAP50_95": DType(metrics.box.map),
                              "precision": DType(metrics.box.mp),
                              "recall": DType(metrics.box.mr),
                              })
    
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    
    # for i in os.listdir(metrics.save_dir):
    #     if i.endswith('.png'):
    #         image = f"{metrics.save_dir}/{i}"
    #         mlflow.log_artifact(local_path=image,run_id=run.info.run_id)
    #     else:
    #          pass
        #mlflow.log_artifact(f"{metrics.save_dir}/F1_curve.png")
        mlflow.log_metric("mAP50", metrics.box.map50)
        mlflow.log_metric("mAP50-95", metrics.box.map)
        mlflow.log_metric("precision", metrics.box.mp)
        mlflow.log_metric("recall", metrics.box.mr)
        mlflow.end_run()
        logger.info("succesfully logged validation metrics")

    
    dataset_root = os.path.split(dataset_config)[0]
    os.makedirs(os.path.join(dataset_root,"data_validation"), exist_ok=True)
    # Compare metrics with the threshold
    if metrics.box.map50 < threshold:
        print(f"model mAP50 {metrics.box.map50} < Threshold value")
        print("Model performance below threshold. Retraining needed.")
        status = "True"
        with open(os.path.join(dataset_root,"data_validation","trigger.txt"), 'w') as file:
            file.write(status)
    else:
        print(f"model mAP50 {metrics.box.map50}>= Threshold value")
        print("Model performance is satisfactory.")
        status = "False"
        with open(os.path.join(dataset_root,"data_validation","trigger.txt"), 'w') as file:
            file.write(status)
    metric = metrics.results_dict
    return status, metric
    





