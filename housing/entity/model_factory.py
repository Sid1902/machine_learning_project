from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
from housing.exception import HousingException
import os
import sys
from collections import namedtuple
from typing import List
from housing.logger import logging
from sklearn.metrics import r2_score,mean_squared_error

# some constants for "model.yaml" file 
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

#some named tuples to get the best model 

BestModel = namedtuple("BestModel",["model_serial_number",
                                    "model",
                                    "best_model",
                                    "best_parameters",
                                    "best_score"])

class ModelFactory :

    def __init__(self,model_config_path :str = None) :
        try:
            self.config : dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module :str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name :str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data :dict = self.config[GRID_SEARCH_KEY][PARAM_KEY]
            self.models_initialization_config :dict = dict(self.config[MODEL_SELECTION_KEY])

            # initialized the model and store them in list
            self.initialized_model_list = None 

            # after performing gridSearch the best parameters and the model will be stored on below list
            self.grid_searched_best_model_list = None 

        except Exception as e:
            raise HousingException(e,sys) from e 
        

    @staticmethod
    def read_params(config_path:str) -> dict :
        try:
            with open (config_path) as yaml_file :
                config :dict = yaml.safe_load(yaml_file)
                return config
        except Exception as e:
            raise HousingException(e,sys) from e 
        

    def get_best_model(self,X,y,base_accuracy = 0.6) -> BestModel :
        logging.info(f"Started initializing model from config file")
        initialized_model_list = self.get_initialized_model_list()
        logging.info(f"Initialized model: {initialized_model_list}")
        

