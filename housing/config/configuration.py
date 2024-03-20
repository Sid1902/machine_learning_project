
# In this Configuration every function is going to return an entity 
from housing.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig, \
ModelTrainingConfig,ModelPusherConfig,ModelEvaluationConfig,TrainingPipelineConfig
import os
from housing.util.util import read_yaml_file

from housing.component import *



class Configuration : 
    
    def __init__(self) -> None:
        pass

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        pass

    def get_data_validation_config(self)-> DataValidationConfig:
        pass

    def get_data_transformation_config(self) ->DataTransformationConfig :
        pass

    def get_model_trainer_config(self) -> ModelTrainingConfig: 
        pass

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        pass

    def get_model_pusher_config(self) -> ModelPusherConfig:
        pass

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        pass