from src.config.configuration import ConfigurationManager
from src.logger import logging
from src.components.c5_prediction import PredictionConfig, Prediction

STAGE_NAME = "Prediction"

class PredictionPipeline:
    def __init__(self):
        pass

    def main(self, q1, q2):
        try:
            logging.info("Entered Method main of PredictionPipeline")
            config = ConfigurationManager()
            prediction_config = config.get_prediction_config()
            prediction = Prediction(config=prediction_config)
            is_duplicate = prediction.predict(q1, q2)
            print(f"Duplicate Questions: {is_duplicate}")
            logging.info("Exited Method main of PredictionPipeline")
        except Exception as e:
            logging.info(f"Error Occured in STAGE {STAGE_NAME}")
            logging.exception(e)
            raise e

q1 = "How can I improve my English speaking skills?"
q2 = "What are the best ways to become fluent in English?"

if __name__ == '__main__':
    logging.info(f"Stage {STAGE_NAME} started")
    prediction_pipeline = PredictionPipeline()
    prediction_pipeline.main(q1, q2)
    logging.info(f"Stage {STAGE_NAME} completed")
