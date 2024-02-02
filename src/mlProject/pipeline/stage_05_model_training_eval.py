from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.stage_05_model_training_eval import Training
from src.mlProject import logger



STAGE_NAME = "Model Training stage"

class DataModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_evaluation_config = config.get_model_train_config()

        # Create an instance of the Training class
        training_instance = Training(data_evaluation_config)

        # Call the train_model method on the instance
        results, model = training_instance.train_model()

        print('Results:', results)
        print('Model:', model)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e




