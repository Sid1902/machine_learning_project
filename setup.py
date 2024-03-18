# It is basically like installing requirements file
# "python setup.py install" and then all libraries needed could be installed 

from setuptools import setup
from typing import List




# Declaring variable for setup function

PROJECT_NAME = "housing-predictor"
VERSION = "0.0.1"
AUTHOR ="Siddhant Bedmutha"
DESCRIPTION = "This is first full stack machine learning project"
PACKAGES = ["housing"]
REQUIREMENT_FILE_NAME = "requirements.txt"



def get_requirements_list()->List[str]:
    """
    Description : This funtion return the list of string of  requirements 
    mentioned in requirements.txt file

    return This function is goung to return a list which contain name 
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file :
        return requirement_file.readlines()



setup(
name= PROJECT_NAME ,# name of the project
version = VERSION,
author = AUTHOR,
description=DESCRIPTION,
packages=PACKAGES,# here provide the name of folder where all packages are present 
install_requires = get_requirements_list(),

)