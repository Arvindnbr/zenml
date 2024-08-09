import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = 'zenml'

list_of_files = [
    ".github/workflows/.gitkeep",
    "scripts/config/__init__.py",
    "scripts/notebook/research.ipynb",
    "scripts/pipeline/__init__.py",
    "scripts/steps/common.py",
    "scripts/utils/__init__.py",
    "scripts/config/configuration.py",
    "scripts/config/__init__.py",
    "config/config.yaml",
    "templates/index.html",
    "main.py",
    "requirements.txt",
    "setup.py",
    ]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as file:
            pass
            logging.info(f"creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} already exists")
                     