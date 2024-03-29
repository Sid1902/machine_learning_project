from housing.exception import HousingException
import sys,os
from housing.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from housing.entity.config_entity import ModelTrainingConfig
from housing.logger import logging
from typing import List
from housing.util.util import load_numpy_array_data,save_object,load_object
from housing.entity.model_factory import ModelFactory,GridSearchBestModel,MetricInfoArtifact
from housing.entity.model_factory import evaluate_regression_model

class HousingEstimatorModel :

    def __init__(self,preprocessing_object,trained_model_object) :
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        
    def predict(self,X) :
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        try :
            transformed_feature = self.preprocessing_object.transform(X)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e :
            raise HousingException(e,sys) from e 
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:

    def __init__(self,model_trainer_config : ModelTrainingConfig,
                data_transformation_artifact : DataTransformationArtifact ):
        try:
            logging.info(f"{'='*20} Model trainer log started {'='*20}")
            self.model_trainer_config  = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e :
            raise HousingException(e,sys) from e 
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact :
        try:
            #  loading transformed train dataset
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_arr = load_numpy_array_data(file_path=transformed_train_file_path)

            #   loading transformed test dataset
            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_arr = load_numpy_array_data(file_path=transformed_test_file_path)

            # Splitting dataset : train and test
            logging.info(f"Splitting training and testing input and target feature ")
            x_train,y_train,x_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            # reading model config file that containes  path to "model.yaml" file
            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing model factory class using model config file {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy : {base_accuracy}")

            # getting best model on traininig dataset
            logging.info(f"Initiating operation model selection")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)

            logging.info(f"best model found on training dataset : {best_model}")

            logging.info(f"Extracting trained model list.")

            grid_searched_best_model_list : List[GridSearchBestModel] = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list ]

            logging.info(f"Evaluating all trained model on training and testing dataset both ")

            #evaluation models on both training & testing datset -->model object
            metric_info : MetricInfoArtifact = evaluate_regression_model(model_list=model_list,
            X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)

            logging.info(f"Best model found on both training and testing dataset")


            #  getting the model name having best accuracy 
            model_object = metric_info.model_object


            # E:\machine_learning_project\machine_learning_project\housing\artifact\data_transformation\2024-03-26-20-13-50\preprocessed\preprocessed.pkl
            #  it is the tranformed data model 

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)

            # E:\machine_learning_project\machine_learning_project\artifact\model_trainer\time_stamp\trianed_model\model.pkl
            trained_model_file_path = self.model_trainer_config.trained_model_file_path

            housing_model = HousingEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)

            logging.info(f"Saving model at path : {trained_model_file_path}")

            save_object(file_path=trained_model_file_path,obj=housing_model)

            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
            trained_model_file_path=trained_model_file_path,
            train_rmse=metric_info.train_rmse,
            test_rmse=metric_info.test_rmse,
            train_accuracy=metric_info.train_accuracy,
            test_accuracy=metric_info.test_accuracy,
            model_accuracy=metric_info.model_accuracy
            
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact      




        except Exception as e:
            raise HousingException(e,sys) from e 
        
    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")


#loading transformed training and testing datset
#reading model config file 
#getting best model on training datset
#evaludation models on both training & testing datset -->model object
#loading preprocessing pbject
#custom model object by combining both preprocessing obj and model obj
#saving custom model object
#return model_trainer_artifact

