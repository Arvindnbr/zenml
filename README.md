# zenml

this is a local MLOPS project that uses zenml and mlflow
## install on local machine

> first `git clone repo` 
> then create an enviornment (python < 3.12 preferablly)
> install the requirements
    '''
     pip install -r requirements.txt 
    '''
> configure zenml

* to register mlflow to zenml localy use;
'''
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
'''
* now set the local store;
'''
zenml artifact-store register local_store --flavor=local --path=/path/to/local/store
'''
* register the orchestrator;
'''
zenml orchestrator register local_orchestrator --flavor=local
'''
* setup the custom stack
'''
zenml stack register custom_stack \
    -e mlflow_experiment_tracker \
    -a local_store \
    -o local_orchestrator \
    --set
'''
* to deploy the model to mlflow
'''
zenml model-deployer register mlflow_deployer --flavor=mlflow
'''


set the changes to the 'config.yaml' file and boom!!!!!

> [!NOTE]
> if there is issue starting zenml use the following commands.
 'zenml status' to check the status of the connection
 'zenml down' to release the connection
 'zenml up' to establish a connection



> [!IMPORTANT]
> to start both mlflow and zenml server before running the code snippet.
 
