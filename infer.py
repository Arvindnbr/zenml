from scripts.entity.exception import AppException
import sys, os
from zenml.logger import get_logger
from zenml import pipeline, log_artifact_metadata, ArtifactConfig
from ultralytics import YOLO
from zenml.client import Client
from scripts.steps.inference import yolov8_prediction, yolov8_validation_step
from scripts.steps.train import load_model, Trainer
from scripts.steps.log_mlflow import register_model
from scripts.steps.best_model import production_model
from scripts.config.configuration import ConfigurationManager
from scripts.steps.retraining import Retrain_dataset, retrain
from zenml import load_artifact
from typing import Annotated, Any, Dict, Tuple
from zenml.materializers.built_in_materializer import BuiltInMaterializer



logger = get_logger(__name__)

dataset = Client().get_artifact_version("Dataset_path")
#val_status_artifact = Client().get_artifact_version("Retrain_trigger")
#model = Client().get_artifact_version("Production_YOLO")

#get dataset artifact
materializer_class = BuiltInMaterializer
dataset_materializer = materializer_class(dataset.uri)
loaded_data = dataset_materializer.load(str)
#status_materializer = materializer_class(val_status_artifact.uri)

config = ConfigurationManager()
threshold = config.get_threshold()
name = config.get_evaluation()
params = config.get_params()
train_config = config.get_train_log_config()
val = config.get_datavalidation_config()

@pipeline
def inference() -> Annotated[str,"trigger status"]:
        
    Dataset = f"{loaded_data}/data.yaml"
    #get model artifact
    model = Client().get_artifact_version("Production_YOLO")
    val_status = yolov8_validation_step(model_path=model,
                                        dataset_config = Dataset,
                                        threshold=threshold.mAP50,
                                        validation_name = name.name
                                        )
    return val_status
    
@pipeline
def Trigger_retrain()-> Annotated[str, ArtifactConfig(name="Retrained_YOLO_model", is_model_artifact=True)]:
    with open(os.path.join(val.root_dir,"Trigger_retrain","trigger.txt"), 'r') as file:
        trigger = file.readline()
    
    if trigger == "False":
        print("Model is healthy!! continue with prediction")
        return model
    elif trigger == "True":
        new_dataset = Retrain_dataset(loaded_data)
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
        #name, version, model = production_model(run_id, threshold)
        return model
    
     




if __name__ == "__main__":
    try:
        inference()
        Trigger_retrain()
    except Exception as e:
        raise AppException(e, sys)
