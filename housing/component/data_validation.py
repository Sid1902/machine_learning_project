from housing.logger import logging
from housing.exception import HousingException
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
import os,sys
from evidently.model_profile import Profile
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
import json



class DataValidation :

    def __init__(self,data_validation_config : DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20} Data Validation log started. {'=' * 20}")
            self.data_validation_config = data_validation_config
            logging.info(f"print {self.data_validation_config}")
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e :
            raise HousingException(e,sys) from e 
        
    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            return train_df,test_df
        except Exception as e:
            raise HousingException(e,sys) from e 
        
    def is_train_test_file_exists(self):
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exists = False
            is_test_file_exists = True

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)

            is_available = is_train_file_exists and is_test_file_exists

            logging.info(f"Is train and test file exists? -> {is_available} ")

            if not is_available :
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message = f"Training file {training_file} or testing file {testing_file} is not present."
                logging.info(message)

                raise Exception(message)

            return is_available
        except Exception as e : 
            raise HousingException(e,sys) from e 
        
    def validate_dataset_schema(self)-> bool :
        try:
            validation_status  = False
            #  what things to validate :
            # 1. no. of columns 
            # 2. check the value of ocean proximity 
            # 3. check column names 







            validation_status = True

            return validation_status 
        except Exception as e:
            raise HousingException(e,sys) from e 
        
    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df,test_df = self.get_train_and_test_df()

            profile.calculate(train_df,test_df)

            profile.json()

            # To convert the json into dictionary format we can use :
            report = json.loads(profile.json())
           
            report_file_path = self.data_validation_config.report_file_path

            report_dir = os.path.dirname(report_file_path)

            os.makedirs(report_dir,exist_ok=True)

            with open(file=report_file_path,mode="w") as report_path :
                json.dump(report,report_path,indent=6)

            return report 
        except Exception as e :
            raise HousingException(e,sys) from e

    def save_data_drift_report_page(self):
        try:
            
            dashboard = Dashboard(tabs=[DataDriftTab()])

            train_df,test_df = self.get_train_and_test_df()

            dashboard.calculate(train_df,test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path

            report_page_dir = os.path.dirname(report_page_file_path)

            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)


            
        except Exception as e :
            raise HousingException(e,sys) from e


        
    def is_data_drift_found(self):
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            

            return True
        except Exception as e :
            raise HousingException(e,sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # before any validation we will check whether the training/testing file exists or not
            self.is_train_test_file_exists()
            self.validate_dataset_schema()

            #  We will use "{EvidentlyAi}" Library to monitor the data drift of the dataset.
            #  for that we need evidently library
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated =True,
                message="Data validation performed successfully."

            )

            logging.info(f"Data Validaiton artifact : {data_validation_artifact} ")
            return data_validation_artifact         
            


        except Exception as e:
            raise HousingException(e,sys) from e 
        

    def __del__(self):
        """
        This code ensures that whenever an object of this class is garbage collected, an informational log message is printed, indicating that the data ingestion process (presumably managed by this class) has been completed
        
        """
        logging.info(f"{'='*20}Data Validation log completed.{'='*20}\n\n")


