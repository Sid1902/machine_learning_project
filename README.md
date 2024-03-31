# machine_learning_project
This is first machine learning project

## start the machine learning project
Application url : 
[CA Price Trends] (https://ml-regression-testing-5830297bf57b.herokuapp.com/)

creating conda environment

```
conda create -p venv python==3.7 -y

```
-p signifies that the virtual environment will be created in the folder itself rather than in anaconda directory.


Activate the conda environment

```
conda activate venv/
```

To install flask
```
pip install flask
```

To add files in git 
```
git add .
```
OR
```
git add <filename>
```

 > To ignore files or folder from git we can write name of file in .gitignore  file


To check the status of git
```
git status 
```

To check all version maintained by git 
```
git log
```

To commit all changes by git 
```
git commit -m "commit_message"
```

To send the commmit/version changes to git
```
git push origin main 
```
To check the remote urls 
```
git remote -v
```


To setup CI/CD pipeline we require 3 info from Heroku 

1. HEROKU_EMAIL = siddhantbedmutha11@gmail.com
2. HEROKU_API_KEY = <>
3. HEROKU_APP_NAME = ml-regression-testing


BUILD DOCKER IMAGE 

```
docker build -t <image_name> :<tag_name> <location of docker_file>
```
> Note : docker image name must be in lowercase and if the location of dockerfile is in same folder then use "." as a location 


To list docker images
```
docker images
```

Run Docker image 
```
docker run - p 5000:5000 -e PORT=5000 <image_id>(ed9cd8d33574)
```

After running image if you want to see wheter the image is working or not check it by typing in browser :
```
localhost:5000
```
To check the running container in docker
```
docker ps 
```

To stop the docker container 
```
docker stop <container_id>
```

> Note : Container id you can use first four characters also.


To install all the libraries and modules :
```
python setup.py install
```


Install ipykernel
```
pip install ipykernel
```


Data Drift : 
Data drift refers to changes in the distribution of the features an ML model receives in production, potentially causing a decline in model performance.

When your dataset stats gets change we call it as data drift.



MLops :
MLOps stands for Machine Learning Operations. MLOps is a core function of Machine Learning engineering, focused on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them. MLOps is a collaborative function, often comprising data scientists, devops engineers, and IT.


Null Hypothesis : Two datsets are from same distribution

alternate Hypothesis : Two datsets are not from same distribution

while re-training our dataset we compared the old and new dataset for their distribution.


Google MLops Docs :
```
https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
```

"""A main.yaml file can serve as a central location to store various 
configuration settings for your project. This can include:
Training and testing parameters for machine learning models 
(e.g., learning rate, batch size, epochs)
Data paths for training, validation, and testing sets Hyperparameter tuning 
options Logging and debugging parameters.
By keeping these configurations in a separate YAML file, you improve code 
readability, maintainability, and the ability to experiment with different 
configurations easily."""