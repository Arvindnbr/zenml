from scripts.entity.exception import AppException
import sys
from zenml.logger import get_logger
from zenml import pipeline
from ultralytics import YOLO
from zenml.client import Client
from scripts.steps.inference import yolov8_prediction, yolov8_validation_step
from scripts.config.configuration import ConfigurationManager

from zenml.materializers.built_in_materializer import BuiltInMaterializer

#use the builtin materializer class


logger = get_logger(__name__)

dataset = Client().get_artifact_version("Dataset_path")
#model = Client().get_artifact_version("Production_YOLO")

materializer_class = BuiltInMaterializer
dataset_materializer = materializer_class(dataset.uri)
#model_materializer = materializer_class(dataset.uri)



@pipeline
def inference():
    config = ConfigurationManager()
    threshold = config.get_threshold()
    name = config.get_evaluation()

    #get dataset artifact
    dataset = Client().get_artifact_version("Dataset_path")
    loaded_data = dataset_materializer.load(str)
    
    Dataset = f"{loaded_data}/data.yaml"

    #get model artifact
    model = Client().get_artifact_version("Production_YOLO")

    status = yolov8_validation_step(model_path=model,
                                    dataset_config = Dataset,
                                    threshold=threshold.mAP50,
                                    validation_name = name.name
                                    )
    if status == False:
        print("initiating model retraining")
        pass
    else:
        print("Model is healthy!! continue with prediction")
    




if __name__ == "__main__":
    try:
        inference()
    except Exception as e:
        raise AppException(e, sys)
