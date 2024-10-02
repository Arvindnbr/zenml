import os, re, random
import yaml
from scripts.utils.log import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any
import base64
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER


ROOT_DIR = Path(__file__).resolve().parents[1]


@ensure_annotations
def read_yaml(yaml_path:Path)-> ConfigBox:
    try:
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_path} loaded sucessfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def update_train_yaml(yaml_path, path_dir):
    with open(yaml_path,'r') as file:
        data = yaml.safe_load(file)
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    new_key = 'path'
    new_value = path_dir
    data[new_key] = new_value

    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data, file,default_flow_style=False)

@ensure_annotations
def create_directories(path_to_dir: list, verbose=True):
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")

@ensure_annotations
def get_size(path:Path) -> str:
    size_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_kb} KB"

@ensure_annotations
def save_json(path:Path, data:dict):
    with open(path,"w") as f:
        content = json.load(f)
    logger.info(f"json file {path} loaded sucessfully")
    return ConfigBox(content)

def decodeImage(imgstr, filename):
    imgdata = base64.b64decode(imgstr)
    with open(filename, 'w') as f:
        f.write(imgdata)
        f.close()

def encodeImage(cropped_imgpath):
    with open (cropped_imgpath, 'rb') as f:
        return base64.b64encode(f.read())
    
#get the train folder from run
def get_highest_train_folder(parent_folder):
    items = os.listdir(parent_folder)
    # Filter the folders and get their numbers
    train_folders = sorted(
        [(item, int(item[5:])) for item in items if os.path.isdir(os.path.join(parent_folder, item)) and re.match(r'train\d+$', item)],
        key=lambda x: x[1],
        reverse=True
    )
    # Iterate over the sorted folders from highest to lowest
    for folder, _ in train_folders:
        weights_path = os.path.join(parent_folder, folder, 'weights', 'best.pt')
        if os.path.exists(weights_path):
            return str(folder)
    
    # Return None if no matching folder contains the weights file
    return None

#copy yaml file

def get_random_file_from_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if files:
        random_file = random.choice(files)
        return os.path.join(folder_path, random_file)
    else:
        print(f"No files found in {folder_path}.")
        return None

#yolo callback

best_loss = float('inf')
wait = 0
patience = 10

def early_stopping_callback(epoch, logs):
            global best_loss, wait
            val_loss = logs.get('val/loss') 
            if val_loss is None:
                print("Validation loss not found in logs.")
                return
            improvement = (best_loss - val_loss) / best_loss * 100
            if improvement < 2:
                wait += 1
            else:
                best_loss = val_loss
                wait = 0
            if wait >= patience:
                print("No improvement, stopping early.")
                model.stop_training = True
            print(f"Epoch {epoch + 1}: Improvement {improvement:.2f}%, Best Loss {best_loss:.4f}")

# def indices(base_set: list, new_set: list):
#     base_set = [item.lower() for item in base_set]
#     new_set = [item.lower() for item in new_set]
#     new_indices =[]
#     for item in new_set:
#         indices = [i for i, x in enumerate(base_set) if x == item]
#         new_indices.extend(indices)
#     return new_indices

def get_images(directory):
    """
    Returns a list of tuples containing the image file names and their full paths
    from the specified directory. It includes both .png and .jpg files.
    
    :param directory: Path to the directory where images are located
    :return: List of tuples (image_name, full_path)
    """
    path = Path(directory)
    image_files = list(path.glob('**/*.[pj][pn]g'))  
    
    images_list = [(image_file.name, str(image_file)) for image_file in image_files]
    
    return images_list

