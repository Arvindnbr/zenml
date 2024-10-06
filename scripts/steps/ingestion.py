import sys, zipfile, os, shutil, yaml
from urllib import request
from zenml import step
from sklearn.model_selection import train_test_split
from scripts.entity.exception import AppException
from scripts.utils.log import logger
from scripts.config.configuration import DataIngestionConfig
from zenml.logger import get_logger
from typing import Annotated
from scripts.utils.common import find_image_file


logger = get_logger(__name__)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def local_data(self):
        images_folder = os.path.join(self.config.data_source, "images")
        labels_folder = os.path.join(self.config.data_source, "labels")
        output_folder = self.config.unzip_dir

        if not os.path.isdir(images_folder) and os.path.isdir(labels_folder):
            raise ValueError("Input folder must contain 'images' and 'labels' subfolder")
        
        filenames = [os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".txt")]

        train, valid = train_test_split(filenames,train_size = 0.7, test_size = 0.3, random_state = 42)
        valid, test = train_test_split(valid, train_size = 0.7, test_size = 0.3, random_state = 42)
        splits = {
            'train': train,
            'valid': valid,
            'test': test
        }

        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)
        for split, file in splits.items():
            for filename in file:
                label_path = os.path.join(labels_folder,f"{filename}.txt")
                if os.path.exists(label_path):
                    image_path = find_image_file(images_folder,filename)
                    if image_path:
                        shutil.copy(image_path,os.path.join(output_folder,split,'images'))
                        shutil.copy(label_path,os.path.join(output_folder,split,'labels'))
                    else:
                        logger.info(f"Image for '{filename}' not found. Skipping.")
                else:
                    logger.info(f"Label file '{filename}.txt' not found. Skipping.")
                
        logger.info(f'Data ingested from {self.config.data_source} to {output_folder}')
        yaml_file = os.path.join(output_folder,"data.yaml")
        yaml_content = {
            'path': f"{output_folder}",
            'train': "train/images",
            'val': "valid/images",
            'test': "test/images",
            'nc': len(self.config.classes),
            'names': self.config.classes
        }
        with open (yaml_file, 'w') as file:
            yaml.dump(yaml_content, file)
        logger.info(f"Dataset created with splits \ntrain: {output_folder}/train \nvalid: {output_folder}/valid \ntest: {output_folder}/test \ncorresponding {yaml_file} created")
        return str(output_folder)
    

    def download_data(self)->str:
        try:
            dataset_url = self.config.data_source
            data_path = f"{self.config.root_dir}/data.zip"
            os.makedirs(self.config.root_dir, exist_ok= True)
            logger.info(f"Downloading data from {dataset_url} into file {data_path}")
            if not os.path.exists(data_path):
                request.urlretrieve(
                    url = dataset_url, filename=data_path
                )
                logger.info(f"{data_path} downloaded!!")
            else:
                logger.info(f"File already exists on {data_path}")
            return str(data_path)
            
        
        except Exception as e:
            raise AppException(e, sys)
    
    def extract_zipfile(self):
        unzip_path = self.config.unzip_dir
        data_path = f"{self.config.root_dir}/data.zip"
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(data_path, 'r') as z:
            z.extractall(unzip_path)
        os.remove(data_path)
        return str(unzip_path)



@step(enable_cache=False)
def data_ingest(config:DataIngestionConfig) -> Annotated[str, "Base_dataset"]:
    try:
        ingestion = DataIngestion(config)
        if os.path.exists(config.data_source):
            directry = str(config.unzip_dir)
            files = [f"{directry}/{split}/{t}" for split in ["train", "valid", "test"] for t in ["images", "labels"]] + [f"{directry}/data.yaml"] + [directry]
            if not any(os.path.exists(path) for path in files):
                return ingestion.local_data()
            else:
                logger.info(f"Dataset pre exist in the specified {config.unzip_dir}\nskipping data ingestion")
                print(f"Dataset pre exist in the specified {config.unzip_dir}\nskipping data ingestion")
                return directry
        else:
            ingestion.download_data()
            return ingestion.extract_zipfile()
    except Exception as e:
        raise AppException(e, sys)

