from collections import namedtuple

# We use namedTuple as it is immutable once it is configured no one can change the names.


DataIngestionConfig = namedtuple("DataIngestionConfig",
["dataset_download_url","tgz_download_dir","raw_data_dir","ingested_train_dir","ingested_test_dir"])


DataValidationConfig = namedtuple("DataValidationConfig",["schema_file_path","report_file_path","report_page_file_path"])


DataTransformationConfig = namedtuple("DataTransformationConfig",[
    "add_bedroom_per_room","transformed_train_dir","transformed_test_dir",
    "preprocessed_object_file_path"
])

ModelTrainingConfig = namedtuple("ModelTrainingConfig",
["trained_model_file_path","base_accuracy","model_config_file_path"])


ModelEvaluationConfig = namedtuple("ModelEvaluationConfig",["model_evaluaiton_file_path","time_stamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig",["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",["artifact_dir"])