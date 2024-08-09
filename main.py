import os
from scripts.pipeline.training_pipeline import data_pipeline
from scripts.config.configuration import ConfigurationManager
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

if __name__ == "__main__":
    x = ConfigurationManager()
    data_pipeline(x.get_dataingestion_config(),
                  x.get_Dataset_config(),
                  x.get_datavalidation_config(),
                  x.get_train_log_config(),
                  x.get_params()
                  )

