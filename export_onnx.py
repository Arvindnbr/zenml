from mlflow.tracking import MlflowClient
from scripts.utils.export import onnx_export
from scripts.config.configuration import ConfigurationManager


#initiate configs
#export model to onnx format
config = ConfigurationManager()
eval = config.get_evaluation()
uri = config.get_train_log_config()
client = MlflowClient(tracking_uri=uri.mlflow_uri)


if __name__ == "__main__":
    
    if not eval.version is None:
        model_version = client.get_model_version(name=eval.name, version=eval.version) 
        run_id = model_version.run_id

        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri.replace("mlflow-artifacts:", "mlartifacts")
        model = f"{artifact_uri}/model/artifacts/best.pt"

        onnx_export(weights=model, output_dir=eval.save_dir, output_name= f"{eval.name}")
        
    else:
        model_versions = client.search_model_versions(f"name = '{eval.name}'")

        production_run_id = None
        for version in model_versions:
            if version.current_stage == "Production":
                production_run_id = version.run_id
                break

        run_info = client.get_run(production_run_id)
        artifact_uri = run_info.info.artifact_uri.replace("mlflow-artifacts:", "mlartifacts")
        model = f"{artifact_uri}/model/artifacts/best.pt"

        onnx_export(weights=model, output_dir=eval.save_dir, output_name= f"{eval.name}")


    
    
    
    