import os
from pathlib import Path
import yaml
from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.components.c2_data_preprocessing import DataPreProcessing

STAGE_NAME = "Data PreProcessing"

class DataPreProcessingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logging.info("Entered Method main of DataPreProcessingPipeline")
            config = ConfigurationManager()
            data_preprocessing_config = config.get_data_preprocessing_config()
            data_preprocessing = DataPreProcessing(data_preprocessing_config)
            data_preprocessing.preprocess()
            logging.info("Exited Method main of DataPreProcessingPipeline")
        except Exception as e:
            logging.info(f"Error Occured in STAGE {STAGE_NAME}")
            logging.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = DataPreProcessingPipeline()
        pipeline.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logging.info(f"Error Occured in STAGE {STAGE_NAME}")
        logging.exception(e)
        raise e