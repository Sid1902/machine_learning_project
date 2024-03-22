from housing.entity.config_entity import DataIngestionConfig
from housing.exception import HousingException
import os,sys
from housing.logger import logging

class DataIngestion :
    def __init__(self,data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20}")
            # ====================Data Ingestion log started.==================== output of {'='*20}
            self.data_ingestion_config = data_ingestion_config
        except Exception as e :
            raise HousingException(e,sys) from e 

    def download_housing_data(self,) :
        pass

    def extract_tgz_file(self,) :
        pass

    def split_data_as_train_test(self,):
        pass

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            pass
        except Exception as e :
            raise HousingException(e,sys) from e 