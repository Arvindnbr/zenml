import os, re, sys
import shutil
import logging
import torch
import random
from ultralytics import YOLO, settings
from sklearn.model_selection import train_test_split
from zenml import step, ArtifactConfig, log_artifact_metadata, save_artifact
from zenml.logger import get_logger
from scripts.entity.exception import AppException
from typing import Annotated, Any, Dict, Tuple
from scripts.config.configuration import ConfigurationManager
from scripts.utils.common import get_highest_train_folder, update_train_yaml
from scripts.config.configuration import Params
from scripts.materializer.yolo_materializer import UltralyticsMaterializer


logger = get_logger(__name__)





class Retrainer:
    def __init__(self, data_dir):
        self.label_dirs = ['train/labels', 'valid/labels', 'test/labels']
        self.data_dir = data_dir

    def versioned_dir(self):
        base_dir = os.path.dirname(self.data_dir)
        current_folder_name = os.path.basename(self.data_dir)
        #check the version
        vmatch = re.match(r'(.+?)_v(\d+)$', current_folder_name)
        
        if vmatch:
            base_name = vmatch.group(1)  
            current = int(vmatch.group(2))  
            new = current + 1
        else:
            base_name = current_folder_name
            new = 1

        newv = f"{base_name}_v{new}"
        new_path = os.path.join(base_dir, newv)
        os.makedirs(new_path, exist_ok=True)
        return new_path
    
    def process_split(self):
        all_filenames = []
        for label_dir in self.label_dirs:
            full_path = os.path.join(self.data_dir, label_dir)
            if os.path.exists(full_path):
                for file_name in os.listdir(full_path):
                    if file_name.endswith('.txt'):
                        base_name = os.path.splitext(file_name)[0]  
                        all_filenames.append(base_name)
    
    
        random.seed(42)
        random.shuffle(all_filenames)
        
        train_filenames, valid_filenames = train_test_split(all_filenames, test_size=0.2, random_state=42)
        
        return train_filenames, valid_filenames
    
    def copy_files(self, file_names, source_dir, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        all_files = os.listdir(source_dir)

        for file_name in file_names:
            matching_files = [f for f in all_files if os.path.splitext(f)[0] == file_name]
            
            for file in matching_files:
                src_file_path = os.path.join(source_dir, file)
                dest_file_path = os.path.join(destination_dir, file)
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")



@step(enable_cache=False)
def Retrain_dataset(data_dir:Annotated[str, "Dataset_path"]
                    )-> Annotated[str, "Dataset_path"]:
    
    retrain = Retrainer(data_dir)
    new_path = retrain.versioned_dir()
    train, valid = retrain.process_split()
    try:
        #train
        retrain.copy_files(train, os.path.join(data_dir,"train/images"), os.path.join(new_path,"train/images"))
        retrain.copy_files(train, os.path.join(data_dir,"valid/images"), os.path.join(new_path,"train/images"))
        retrain.copy_files(train, os.path.join(data_dir,"test/images"), os.path.join(new_path,"train/images"))
        retrain.copy_files(train, os.path.join(data_dir,"train/labels"), os.path.join(new_path,"train/labels"))
        retrain.copy_files(train, os.path.join(data_dir,"valid/labels"), os.path.join(new_path,"train/labels"))
        retrain.copy_files(train, os.path.join(data_dir,"test/labels"), os.path.join(new_path,"train/labels"))
        #valid
        retrain.copy_files(valid, os.path.join(data_dir,"train/images"), os.path.join(new_path,"valid/images"))
        retrain.copy_files(valid, os.path.join(data_dir,"valid/images"), os.path.join(new_path,"valid/images"))
        retrain.copy_files(valid, os.path.join(data_dir,"test/images"), os.path.join(new_path,"valid/images"))
        retrain.copy_files(valid, os.path.join(data_dir,"train/labels"), os.path.join(new_path,"valid/labels"))
        retrain.copy_files(valid, os.path.join(data_dir,"valid/labels"), os.path.join(new_path,"valid/labels"))
        retrain.copy_files(valid, os.path.join(data_dir,"test/labels"), os.path.join(new_path,"valid/labels"))
        #test folder
        os.makedirs(os.path.join(new_path,"test","images"), exist_ok=True)
        os.makedirs(os.path.join(new_path,"test","labels"), exist_ok=True)
        #yaml file
        shutil.copyfile(os.path.join(data_dir,"data.yaml"),os.path.join(new_path,"data.yaml"))
        update_train_yaml(os.path.join(new_path,"data.yaml"),new_path)

    except Exception as e:
        raise AppException(e, sys)
    
    save_artifact(new_path,"Dataset_path")
    return new_path


@step(output_materializers={"Trained_YOLO": UltralyticsMaterializer}, enable_cache=False)
def retrain(config: Params, 
            dataset: Annotated[str, "Dataset_path"], 
            yolo_model: YOLO) -> tuple[Annotated[YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)],
                                       Annotated[Dict[str, Any], "validation_metrics"],
                                       Annotated[Dict[str, Any], "model_names"],
                                       Annotated[str, "Model metrics"]]:
    settings.update({'mlflow': True})
    settings.reset()

    data_path = os.path.join( dataset, "data.yaml")
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

    trained = model.train(
                data=data_path,
                optimizer = config.optimizer,
                lr0 = config.lr0,
                save_period = config.save_period,
                batch = config.batch,
                epochs = config.epochs,
                resume = config.resume,
                seed = config.seed,
                imgsz = config.imgsz,
                patience = 10)
    train_folder = get_highest_train_folder("runs/detect")
    save_dir = f"runs/detect/{train_folder}"
    log_artifact_metadata(artifact_name="Trained_YOLO",
                          metadata={"metrics": trained.results_dict, 
                                    "names": model.names, 
                                    "Save directory": save_dir},)
    
    return model, trained.results_dict, model.names, save_dir