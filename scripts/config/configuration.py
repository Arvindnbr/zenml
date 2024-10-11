
from scripts.utils.common import read_yaml, create_directories, load_params_from_yaml
from scripts.constant import *
from scripts.entity.entity import *



class ConfigurationManager:
    def __init__(self,config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.configparam = load_params_from_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_dataingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            data_source= config.data_source,
            unzip_dir= config.unzip_dir,
            classes = config.classes
            )
        return data_ingestion_config
    
        #optional
    def get_Dataset_config(self)-> DataSetConfig:
        config = self.config.custom_dataset
        dataset_config = DataSetConfig(
            classes= config.classes,
            new_data_path= config.new_data_path,
            dataset_name=config.dataset_name
        )
        return dataset_config
    
    
    def get_train_log_config(self)-> TrainLogConfig:
        config = self.config.train_log_config
        trainlogconfig = TrainLogConfig(
            model= config.model,
            mlflow_uri= config.mlflow_uri,
            experiment_name= config.experiment_name,
            model_name= config.model_name,
            runs_root= self.config.artifacts_root
            )
        return trainlogconfig
    
    def get_params(self)-> Params:
        param = self.configparam
        params = Params(
            optimizer = param.optimizer,
            lr0 = param.lr0,
            save_period = param.save_period,
            batch = param.batch,
            epochs = param.epochs,
            resume = param.resume,
            seed = param.seed,
            imgsz = param.imgsz,
            patience = param.patience,
            cache = param.cache, 	       
            workers = param.workers,	         
            cos_lr = param.cos_lr,	     
            close_mosaic = param.close_mosaic, 	   
            amp = param.amp,
            lrf = param.lrf, 	         
            momentum = param.momentum, 	   
            weight_decay = param.weight_decay, 
            warmup_epochs = param.warmup_epochs,
            warmup_momentum = param.warmup_momentum,
            warmup_bias_lr = param.warmup_bias_lr,
            box = param.box,
            cls = param.cls,
            hsv_h = param.hsv_h,               
            hsv_s = param.hsv_s,                  
            hsv_v = param.hsv_v,                
            degrees = param.degrees,               
            translate = param.translate,           
            scale = param.scale,        
            shear = param.shear,               
            perspective = param.perspective,           
            flipud = param.flipud,                
            fliplr = param.fliplr,           
            bgr = param.bgr,             
            mosaic = param.mosaic,                 
            mixup = param.mosaic,             
            copy_paste = param.copy_paste,             
            auto_augment = param.auto_augment, 
            erasing = param.erasing,     
            crop_fraction = param.crop_fraction,
            )
        return params
    
    def get_threshold(self)-> TresholdMetrics:
        threshold = self.config.threshold
        metrics = TresholdMetrics(
            mAP50= threshold.mAP50,
            mAP50_95= threshold.mAP50_95
        )
        return metrics
    
    def get_evaluation(self)-> Evaluation:
        eval = self.config.evaluation
        evals = Evaluation(
            name = eval.name,
            version = eval.version,
            data_source= eval.data_source,
            save_dir= eval.save_dir
        )
        return evals



