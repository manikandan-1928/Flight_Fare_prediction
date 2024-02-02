from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.stage_02_data_cleaning import DataCleaning
from src.mlProject import logger



STAGE_NAME = "Data Cleaning stage"

class DataCleaningTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        data_cleaning_process = DataCleaning(config=data_cleaning_config)
        data_cleaning_process.data_cleaning()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataCleaningTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

