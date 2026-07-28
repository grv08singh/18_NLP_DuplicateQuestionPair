from src.config.configuration import ConfigurationManager
from src.components.c1_data_ingestion import DataIngestion
from src.logger import logging

STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logging.info("Entered Method main of DataIngestionPipeline")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.create_raw_data()
            logging.info("Exited Method main of DataIngestionPipeline")
        except Exception as e:
            logging.info("Error occured in method main of DataIngestionPipeline")
            logging.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionPipeline()
        pipeline.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logging.info(f"Error Occured in STAGE {STAGE_NAME}")
        logging.exception(e)
        raise e