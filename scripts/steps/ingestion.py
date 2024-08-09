# data_ingestion_step.py
import sys, zipfile, os
from urllib import request
from zenml import step
from scripts.entity.exception import AppException
from scripts.utils.log import logger
from scripts.config.configuration import DataIngestionConfig
from zenml.logger import get_logger
from typing import Annotated, Any, Dict, Tuple

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    

    def download_data(self)->str:
        try:
            dataset_url = self.config.source_URL
            os.makedirs(self.config.root_dir, exist_ok= True)
            logger.info(f"Downloading data from {dataset_url} into file {self.config.local_data}")
            if not os.path.exists(self.config.local_data):
                request.urlretrieve(
                    url = dataset_url, filename=self.config.local_data
                )
                logger.info(f"{self.config.local_data} downloaded!!")
            else:
                logger.info(f"File already exists on {self.config.local_data}")
            return str(self.config.local_data)
            
        
        except Exception as e:
            raise AppException(e, sys)
    
    def extract_zipfile(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data, 'r') as z:
            z.extractall(unzip_path)
        os.remove(self.config.local_data)
        return str(unzip_path)



@step(enable_cache=True)
def data_ingest(config:DataIngestionConfig) -> Annotated[str, "Base_dataset"]:
    try:
        ingestion = DataIngestion(config)
        ingestion.download_data()
        return ingestion.extract_zipfile()
    except Exception as e:
        raise AppException(e, sys)

