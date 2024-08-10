from zenml import pipeline
import os
from scripts.entity.exception import AppException
from scripts.steps.ingestion import data_ingest
from scripts.steps.sorting import data_sort
from scripts.steps.validation import validator
from scripts.steps.train import Trainer, load_model
from scripts.steps.log_mlflow import register_model
from scripts.config.configuration import ConfigurationManager
from scripts.entity.entity import DataIngestionConfig, DataSetConfig, DataValidationConfig, TrainLogConfig, Params, TresholdMetrics
from scripts.steps.best_model import production_model



@pipeline
def data_pipeline(config: DataIngestionConfig, 
                  dset_config: DataSetConfig, 
                  val_config: DataValidationConfig,
                  trainlog_config: TrainLogConfig, 
                  parameters: Params,
                  threshold: TresholdMetrics 
                  ):
    datapath = data_ingest(config)
    print(datapath)
    config1 = ConfigurationManager()
    dsort = config1.get_Dataset_config()

    if len(dsort.classes) != 0:
        current_dset = data_sort(dset_config, datapath)
        print(current_dset)
    else:
        current_dset = datapath
        print(current_dset)

    status = validator(val_config,current_dset)
    yolo_model = load_model(trainlog_config.model)
    yolo_model, metrics, names, save_dir = Trainer(config=trainlog_config,
                                                   val_config=val_config,
                                                   params=parameters,
                                                   validation_status=status,
                                                   current_dset=current_dset,
                                                   yolo_model=yolo_model
                                                   )                                            
    os.environ["MLFLOW_TRACKING_URI"] = trainlog_config.mlflow_uri
    run_id, experiment_id = register_model(experiment_name = trainlog_config.experiment_name,
                                           model_name = trainlog_config.model_name,
                                           save_dir = save_dir
                                           )
    production_model(run_id, threshold)



    
    