from zenml import step
from zenml.logger import get_logger
import os,sys
from scripts.utils.log import logger
from scripts.entity.exception import AppException
from scripts.utils.common import update_train_yaml
from scripts.config.configuration import DataValidationConfig
from typing import Annotated, Any, Dict, Tuple

logger = get_logger(__name__)


class DataValidation:
    def __init__(self, config: DataValidationConfig, current_path: str):
        self.config = config
        self.current_path = current_path

    def validate_files(self, current_path)-> bool:
        try:
            validation_status = None
            all_files = os.listdir(current_path)
            os.makedirs(self.config.root_dir, exist_ok=True)
            status_file = f"{self.config.root_dir}/status.txt"
            for file in all_files:
                if file not in self.config.req_files:
                    validation_status = False
                    with open(status_file,'w') as f:
                        f.write(f"validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(status_file, 'w') as f:
                        f.write(f"validation_status: {validation_status}")
            print(all_files)
            return validation_status
        except Exception as e:
            raise AppException(e,sys)
        
    def update_yaml(self, current_path):
        yamlpath = os.path.join(current_path,"data.yaml")
        update_train_yaml(yamlpath,current_path)
        logger.info(f"following changes has been made \n train and valid path inside data.yaml has been modified \n path : {current_path} has been added to data.yaml file ")

    


@step(enable_cache=False)
def validator(config:DataValidationConfig, dir:str)->Annotated[bool, "validation status"]:
    try:
        validator = DataValidation(config, dir)
        status = validator.validate_files(dir)
        validator.update_yaml(dir)
        return status
    except Exception as e:
        raise AppException(e,sys)

















    