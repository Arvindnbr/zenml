from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataSetConfig:
    classes: list
    new_data_path: Path
    dataset_name: str

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    req_files: list

@dataclass(frozen=True)
class TrainLogConfig:
    model: str
    mlflow_uri: str
    experiment_name: str
    model_name: str

@dataclass(frozen=True)
class Params:
    optimizer: str
    lr0: float
    save_period: int
    batch: int
    epochs: int
    resume: bool
    seed: int
    imgsz: int 

@dataclass(frozen=True)
class TresholdMetrics:
    mAP50: float
    mAP50_95: float