# It is basically like installing requirements file
# "python setup.py install" and then all libraries needed could be installed 

from setuptools import setup,find_packages
from typing import List


# Declaring variable for setup function

PROJECT_NAME = "housing-predictor"
VERSION = "0.0.2"
AUTHOR ="Siddhant Bedmutha"
AUTHOR_EMAIL = "siddhantbedmutha111@gmail.com"
DESCRIPTION = "This is first full stack machine learning project"
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
packages=find_packages(),# return all the folder name where "__init__" and treat them as package
install_requires = get_requirements_list(),
author_email=AUTHOR_EMAIL

)