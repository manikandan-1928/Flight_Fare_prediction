from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.stage_04_data_transformation import DataTransformation
from src.mlProject import logger



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_trans_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_trans_config)
        data_transformation.initiate_data_transformation()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e