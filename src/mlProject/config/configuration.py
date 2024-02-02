from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories
from src.mlProject.entity.config_entity import (DataIngestionConfig,DataCleaningConfig,DataValidationConfig,
                                                DataTransformationConfig,DataModelConfig,DataTuningConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning

        create_directories([config.root_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            validated_path=config.validated_path
        )

        return data_cleaning_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            preprocessor_obj_file_path=config.preprocessor_obj_file_path
        )

        return data_transformation_config
    

    def get_data_model_config(self) -> DataModelConfig:

        print('Done')
        config = self.config.data_model

        create_directories([config.root_dir])

        data_model_config = DataModelConfig(
            root_dir=config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
        )
        return data_model_config
    


    def get_model_train_config(self) -> DataTuningConfig:

        config = self.config.evaluate_model
        params = self.params

        create_directories([config.root_dir])

        data_train_config = DataTuningConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path = config.test_data_path,
            params_report = config.params_report,
            results_report = config.results_report,
            model_report = config.model_report,
            all_params = params,
            mlflow_uri= 'https://dagshub.com/manikandan-1928/Flight_Fare_prediction.mlflow'
        )

        return data_train_config
    


    






    

