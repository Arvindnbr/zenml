from zenml.logger import get_logger
import mlflow, os
from typing import Annotated, Any, Dict, Tuple
from zenml import step, log_artifact_metadata, ArtifactConfig, save_artifact
from ultralytics import YOLO
from scripts.entity.entity import Evaluation
from zenml.metadata.metadata_types import StorageSize, DType
from scripts.config.configuration import ConfigurationManager
from pathlib import Path


logger = get_logger(__name__)


@step(enable_cache=False)
def yolov8_prediction(model_path: str,
                      image_source: str,
                      dir: str):
    """Performs prediction using a YOLOv8 model and saves the results.

    Args:
        model_path (str): Path to the YOLO model.
        image_source (str): Path to the image or folder of images.
        name (str): Name of the subfolder to save the results in.
    """
    predictions = []
    model = YOLO(model_path)
    project= "data/evaluation/preidction"
    results = model(source=image_source, save=True, project=project, name=dir, save_txt = True)

    for r in results:
        prediction_details = {
            "image_path": r.path,
            "boxes": r.boxes.xyxy.cpu().numpy().tolist(),  # bounding boxes
            "confidences": r.boxes.conf.cpu().numpy().tolist(),  # confidence scores
            "classes": r.boxes.cls.cpu().numpy().tolist(),  # class indices
            }
        predictions.append(prediction_details)
        #x = r.show()
        # log_artifact_metadata(artifact_name=f"predicted_{r.path}",
        #                       metadata={
        #                           "image_path": DType(r.path),
        #                           "boxes": DType(r.boxes.xyxy.cpu().numpy().tolist()),  # bounding boxes
        #                           "confidences": DType(r.boxes.conf.cpu().numpy().tolist()),  # confidence scores
        #                           "classes": DType(r.boxes.cls.cpu().numpy().tolist()),
        #                           })
    return predictions
            



@step(enable_cache=False)
def yolov8_validation_step(model_path: str,
                           dataset_config: str,
                           threshold: float,
                           validation_name:str) -> Annotated[str, "Retrain_trigger"]:
           #Annotated[Dict[str, Any], ArtifactConfig(name="infered_metrics",is_model_artifact=True)]]:
    """Validates the YOLOv8 model and checks if retraining is needed.

    Args:
        model_path (str): Path to the YOLO model.
        dataset_config (str): Path to the dataset configuration file.
        threshold (float): Threshold value for performance metrics.
        validation_project (str): Path to the project folder where validation results will be saved.
        validation_name (str): Name of the subfolder to save validation results.

    Returns:
        bool: True if retraining is needed, False otherwise.
    """
    model = YOLO(model_path)
    validation_project ="data/evaluation/inference"
    metrics = model.val(data=dataset_config, split="test", project=validation_project, name=validation_name)
    print(f"mAP50 of model : {metrics.box.map50}")
    print(f"mAP50-95 of model : {metrics.box.map}")

    log_artifact_metadata(artifact_name="infered_metrics",
                          metadata={
                              "mAP50": DType(metrics.box.map50),
                              "mAP50_95": DType(metrics.box.map),
                              "precision": DType(metrics.box.mp),
                              "recall": DType(metrics.box.mr),
                              })
    
    x = ConfigurationManager()
    trigger_dir = x.get_datavalidation_config()
    os.makedirs(os.path.join(trigger_dir.root_dir,"Trigger_retrain"), exist_ok=True)
    # Compare metrics with the threshold
    if metrics.box.map50 < threshold:
        print(f"model mAP50 {metrics.box.map50} < Threshold value")
        print("Model performance below threshold. Retraining needed.")
        status = "True"
        with open(os.path.join(trigger_dir.root_dir,"Trigger_retrain","trigger.txt"), 'w') as file:
            file.write(status)
    else:
        print(f"model mAP50 {metrics.box.map50}>= Threshold value")
        print("Model performance is satisfactory.")
        status = "False"
        with open(os.path.join(trigger_dir.root_dir,"Trigger_retrain","trigger.txt"), 'w') as file:
            file.write(status)
    
    return status
    





