from typing import Annotated, Any, Dict, Tuple
import mlflow
import logging
from zenml import get_step_context, step
from dataclasses import asdict
from zenml.logger import get_logger
from scripts.config.configuration import TresholdMetrics

logger = get_logger(__name__)



class ProductionModel:
    def __init__(self) -> None:
        self.client = mlflow.tracking.MlflowClient()

    def get_registered_model_name_from_run(self,run_id: str) -> str:
        # Search for all registered models
        registered_models = self.client.search_registered_models()

        # Iterate through all registered models
        for model in registered_models:
            # Get all versions of the model
            for version in model.latest_versions:
                # Check if the run ID matches the input run_id
                if version.run_id == run_id:
                    logging.info(f"{model.name} is registered under {run_id}")
                    return model.name
                    
        
        # If no matching model is found, return None or an appropriate message
        return None

    def list_models_and_versions(self, name=None):
        # Get all registered models
        if name:
            registered_models = self.client.search_registered_models(filter_string="name LIKE '%'")
        
        models_info = {}

        for model in registered_models:
            model_name = model.name
            models_info[model_name] = []

            # Get all versions of the registered model
            versions = model.latest_versions
            
            for version in versions:
                version_info = {
                    "version": version.version,
                    "run_id": version.run_id,
                    "source": version.source
                }
                models_info[model_name].append(version_info)

        logging.info("model info has been fetched")
        return models_info

    def get_model_metrics(self, run_id):
        
        run_data = self.client.get_run(run_id).data.to_dictionary()
        metrics = {
            "mAP50": run_data['metrics'].get('mAP50'),
            "mAP50_95": run_data['metrics'].get('mAP50-95'),
            # "recall": run_data['metrics'].get('recall'),
            # "precision": run_data['metrics'].get('precision')
        }
        logging.info(f"metrics of {run_id} has been retrieved")
        return metrics
    

@step   
def production_model(run_id: str,  
                     thresholds:TresholdMetrics) -> Tuple[Annotated[str, "Production model"],
                                                          Annotated[int, "version"]]:

    client = mlflow.tracking.MlflowClient()
    prod = ProductionModel()
    name = prod.get_registered_model_name_from_run(run_id=run_id)
    models_info = prod.list_models_and_versions(name)
        
    if name not in models_info:
        return "Model not found."
        
    best_run_id = None
    best_metrics = None
    best_version = None
       
    for version_info in models_info[name]:
        run_id = version_info['run_id']
        metrics = prod.get_model_metrics(run_id)

        threshold_dict = asdict(thresholds)
            
        if all(metrics[metric] >= threshold_dict[metric] for metric in threshold_dict):
            if best_metrics is None or (
                metrics['mAP50_95'] > best_metrics['mAP50_95'] or
                (metrics['mAP50_95'] == best_metrics['mAP50_95'] and metrics['mAP50'] > best_metrics['mAP50'])
            ):
                best_run_id = run_id
                best_metrics = metrics
                best_version = int(version_info['version'])
        
    if best_run_id is not None:
        print(f"Best version: {best_version}")
        print(f"Best run ID: {best_run_id}")
        print(f"Best metrics: {best_metrics}")
        
        client.transition_model_version_stage(
            name=name,
            version=best_version,
            stage="Production",
            archive_existing_versions=True  # Optional: Archives previous versions in Production
        )
        logging.info(f"{name} model with version{best_version} has been set to production")
        return name, best_version
    else:
        logging.info("models failed to meet the threshold, initiate retrain")
        return tuple("No models meet the threshold criteria. kindly update parameters and retrian")