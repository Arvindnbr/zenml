from zenml import pipeline
from scripts.steps.best_model import production_model 
from scripts.entity.entity import TresholdMetrics
from ultralytics import YOLO
from zenml.client import Client
import sys
from scripts.steps.inference import yolov8_prediction, yolov8_validation_step
from scripts.config.configuration import ConfigurationManager
from scripts.entity.exception import AppException

@pipeline
def prediction(image_source: str):
    config = ConfigurationManager()
    eval = config.get_evaluation()
    model = Client().get_artifact_version("Production_YOLO")

    yolov8_prediction(model_path=model,
                      image_source=image_source,
                      dir = eval.name)




if __name__ == "__main__":
    try:
        prediction("MLartifacts/data_ingestion/v2/images")
    except Exception as e:
        raise AppException(e, sys)
    