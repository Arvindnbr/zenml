from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_source: str
    unzip_dir: Path
    classes: list

@dataclass(frozen=True)
class DataSetConfig:
    classes: list
    new_data_path: Path
    dataset_name: str

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

@dataclass(frozen=True)
class Evaluation:
    name: str