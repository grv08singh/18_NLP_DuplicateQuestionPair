from src.config.configuration import ConfigurationManager
from src.logger import logging
from src.components.c4_model_training import Training

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info("Entered Method main of ModelTrainingPipeline")
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.train()
            logging.info("Exited Method main of ModelTrainingPipeline")
        except Exception as e:
            logging.info(f"Error Occured in STAGE {STAGE_NAME}")
            logging.exception(e)
            raise e

if __name__ == '__main__':
    logging.info(f"Stage {STAGE_NAME} started")
    training_pipeline = ModelTrainingPipeline()
    training_pipeline.main()
    logging.info(f"Stage {STAGE_NAME} completed")
