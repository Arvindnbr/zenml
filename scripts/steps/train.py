import os, sys
import logging
import torch
from scripts.entity.exception import AppException
from scripts.config.configuration import *
from ultralytics import YOLO, settings
from zenml import step
from zenml.logger import get_logger
from scripts.materializer.yolo_materializer import UltralyticsMaterializer
from scripts.utils.common import get_highest_train_folder

from typing import Annotated, Any, Dict, Tuple
from ultralytics import YOLO
from zenml import ArtifactConfig, step, log_artifact_metadata



logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config: TrainLogConfig,val: DataValidationConfig,param: Params):
        self.config = config
        self.param = param
        self.val = val

    def validation_status(self, status):
        status_file = f"{self.val.root_dir}/status.txt"
        with open(status_file, 'r') as file:
            status = file.read().strip()
        key, value = status.split(':')
        key = key.strip()
        value = value.strip()

        if key != "validation_status":
            raise ValueError("unexpected key in status file")
        
        if value == 'True':
            if value == status:
                return True
        elif value == 'False':
            return False
        else:
            raise ValueError("validation status is invalid")
        

    def train_model(self, current_dset, yolo_model):

        settings.update({'mlflow': True})
        settings.reset()

        dataset_dir = current_dset
        data_path = os.path.join( dataset_dir, "data.yaml")
        logging.info(f"Dataset location: {data_path}")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Device is running on: {torch.cuda.get_device_name(device)}")
        else:
            logging.info(f"CUDA is not available")
            device = "cpu"
            logging.info(f"Device to run on: {device}")
            logging.info(data_path)

        model = yolo_model
        # Load a pretrained YOLOv8n model
        # model = YOLO(model)
        # Train the model
        trained = model.train(
                data=data_path,
                optimizer = self.param.optimizer,
                lr0 = self.param.lr0,
                save_period = self.param.save_period,
                batch = self.param.batch,
                epochs = self.param.epochs,
                resume = self.param.resume,
                seed = self.param.seed,
                imgsz = self.param.imgsz
                )
        # evaluate model performance on the validation set
    
        return trained
        


@step(enable_cache=True, enable_step_logs=False)
def load_model(model_checkpoint: str,
               ) -> Annotated[YOLO, ArtifactConfig(name="Raw_YOLO", is_model_artifact=True)]:
    logger.info(f"Loading YOLO checkpoint {model_checkpoint}")
    return YOLO(model_checkpoint)    

@step(output_materializers={"Trained_YOLO": UltralyticsMaterializer},
      enable_cache=True,
      )
def Trainer(config:TrainLogConfig,val_config: DataValidationConfig,
            params:Params,validation_status: bool,current_dset: str,
            yolo_model: YOLO) -> Tuple[Annotated[YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)],
                                       Annotated[Dict[str, Any], "validation_metrics"],
                                       Annotated[Dict[str, Any], "model_names"],
                                       Annotated[str, "Model metrics"]]:
    try:
        trainer = ModelTrainer(config,val_config,params)
        trainer.validation_status(validation_status)
        trained = trainer.train_model(current_dset, yolo_model)
        train_folder = get_highest_train_folder("runs/detect")
        save_dir = f"runs/detect/{train_folder}"
        log_artifact_metadata(artifact_name="Trained_YOLO",metadata={"metrics": trained.results_dict, "names": yolo_model.names, "Save directory": save_dir},)
        return yolo_model, trained.results_dict, yolo_model.names,save_dir
    except Exception as e:
        raise AppException(e, sys)
    
