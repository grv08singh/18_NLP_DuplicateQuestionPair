from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.components.c3_data_processing import DataProcessing

STAGE_NAME = "Data Processing"

class DataProcessingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logging.info("Entered Method main of DataProcessingPipeline")
            config = ConfigurationManager()
            data_processing_config = config.get_data_processing_config()
            data_processing = DataProcessing(data_processing_config)
            data_processing.Process()
            logging.info("Exited Method main of DataProcessingPipeline")
        except Exception as e:
            logging.info(f"Error Occured in STAGE {STAGE_NAME}")
            logging.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = DataProcessingPipeline()
        pipeline.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logging.info(f"Error Occured in STAGE {STAGE_NAME}")
        logging.exception(e)
        raise e