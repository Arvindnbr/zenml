# zenml


try to install zenml to the zenml directory rather than the scripts part
then install zenml[server]

then to register mlflow to zenml localy use;
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

now set the local store;
zenml artifact-store register local_store --flavor=local --path=/path/to/local/store

register the orchestrator;
zenml orchestrator register local_orchestrator --flavor=local

setup the custom stack
zenml stack register custom_stack \
    -e mlflow_experiment_tracker \
    -a local_store \
    -o local_orchestrator \
    --set

to deploy the model to mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow