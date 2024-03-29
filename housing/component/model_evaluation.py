from housing.exception import HousingException
from housing.logger import logging
from housing.constant import *
from housing.entity.artifact_entity import ModelTrainerArtifact,DataValidationArtifact,DataIngestionArtifact,ModelEvaluationArtifact
from housing.entity.config_entity import ModelEvaluationConfig
import numpy as np
import os 
import sys
from housing.util.util import read_yaml_file,load_object,load_data,write_yaml_file
from housing.entity.model_factory import evaluate_regression_model

"""considering this as for Re-training also Model evaluation here means comparing 
the accuracty between the newly trained model and model 
that is already present in production""" 


class ModelEvaluation :

    def __init__(self,model_evaluation_config : ModelEvaluationConfig,
                 data_ingestion_artifact : DataIngestionArtifact,
                 data_validation_artifact : DataValidationArtifact,
                 model_trainer_artifact : ModelTrainerArtifact) :
        
        try:

            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e :
            raise HousingException(e,sys) from e 
        
    def get_best_model(self):
        try :
            model = None

            model_evaluation_file_path = self.model_evaluation_config.model_evaluaiton_file_path

            # if the directory doesnt exist then create empty "model_evaluation.yaml" 
            # and return model which means previously there's no model present
            if not os.path.exists(model_evaluation_file_path) :
                write_yaml_file(file_path=model_evaluation_file_path,)

                return model
            
            #  and if file exists then try to read the content 
            model_eval_file_content =read_yaml_file(file_path=model_evaluation_file_path)
            
            # if file exists but has no content then make empty dict or else read the content present
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            # if "best_model" key not present in eval_content then :
            if BEST_MODEL_KEY not in model_eval_file_content :
                return model # this means no model present 
            
            #  but if "best_model" is present in file then load the model present in that path 
                
            # This would be best model currently present in production

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])

            return model 
        
        except Exception as e :
            raise HousingException(e,sys) from e 
        

    def update_evaluation_report(self,model_evaluation_artifact : ModelEvaluationArtifact) :
        try :
            eval_file_path = self.model_evaluation_config.model_evaluaiton_file_path

            model_eval_content = read_yaml_file(file_path=eval_file_path)

            # if file exists but has no content then make empty dict or else read the content present
            model_eval_content = dict() if model_eval_content is None else model_eval_content

            previous_best_model = None 
            if BEST_MODEL_KEY in model_eval_content :
                previous_best_model = model_eval_content[BEST_MODEL_KEY] 
            logging.info(f"Previous evaluation result :{previous_best_model}")

            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }



            if previous_best_model is not None :
                # if there is any previous model path then we need to create history versions
                # i.e we need to update the new model path and store the previous under history tag 
                #  this will happen only when the newly_trained model is better than the prodcution model

                model_history = {self.model_evaluation_config.time_stamp : previous_best_model}
                # creating key value pair like : {'2024-03-29' : previous_model_path}
                if HISTORY_KEY not in model_eval_content :
                    history = {HISTORY_KEY : model_history}
                    eval_result.update(history)

                else :
                    model_eval_content[HISTORY_KEY].update(model_history)

                    #a=dict()
                    #a
                    # {}
                    #new_A = {'data':1}
                    #a.update(new_A)
                    #a
                    # {'data': 1}
                    #new_data = {"data":2}
                    #a
                    # {'data': 1}
                    #a.update(new_data)
                    #a
                    # {'data': 2}
                    #history={""}
                    #history
                    # {''}
                    #history={"time_stamp":{"data":1}} 
                    #a.update(history)
                    #a
                    # {'data': 2, 'time_stamp': {'data': 1}}
                    #  this is how dict operations will happen 
            
            model_eval_content.update(eval_result) 
            # if previous best model is none then just asign  eval_result declared on line 80
            logging.info(f"updated eval result :{model_eval_content}")
            write_yaml_file(file_path=eval_file_path,data=model_eval_content)               



        except Exception as e :
            raise HousingException(e,sys) from e 
        
        

    def initiate_model_evaluation (self) -> ModelEvaluationArtifact :
        try :
            # getting the trained model path 
            # E:\machine_learning_project\machine_learning_project\housing\artifact\model_trainer\2024-03-29-17-07-12\trained_model\model.pkl
            trained_model_file_path  = self.model_trainer_artifact.trained_model_file_path

            #  getting the model using path 
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            # loading training and testing file 
            train_dataframe = load_data(file_path=train_file_path,schema_file_path=schema_file_path)

            test_dataframe = load_data(file_path=test_file_path,schema_file_path=schema_file_path)

            # finding the target column from schema file 
            schema_content  = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target column ----> Y
            logging.info(f"Converting target column into numpy array ")
            train_target_arr = np.array(train_dataframe[target_column_name])

            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conersion completed targeted column into numpy array")

            # dropping target column from the dataframe -----> X
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)

            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            #  We created the train and test array in order to evaluate the previously present model and newly trained model

            model = self.get_best_model()

            if model is None :
                logging.info("not found any existing model . Hence accepting trained model ")
                #  this case occurs only when model is trained for the first time 

                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                

                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            # if the model is not none then we need to compare :
            # previosuly present model and newly trained model 
            model_list = [model,trained_model_object]

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                               X_train=train_dataframe,
                                                               y_train=train_target_arr,
                                                               X_test=test_dataframe,
                                                               y_test=test_target_arr,
                                                               base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
            
            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact

        except Exception as e :
            raise HousingException(e,sys) from e 
        

    def __del__(self) :
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} \n \n ")