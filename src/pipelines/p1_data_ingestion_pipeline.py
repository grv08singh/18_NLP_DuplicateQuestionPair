from src.config.configuration import ConfigurationManager
from src.components.c1_data_ingestion import DataIngestion
from src.logger import loggging

STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            logging.info("Entered main method of DataIngestionPipeline")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.copy_file()
            logging.info("Exited main method of DataIngestionPipeline\n\n")
        except Exception as e:
            logging.info(f"Data Ingestion failed")
            logging.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logging.info(f"Stage Started: {STAGE_NAME}")
        pipeline = DataIngestionPipeline()
        pipeline.main()
        logging.info(f"Stage Completed: {STAGE_NAME}")
    except Exception as e:
        logging.info(f"{STAGE_NAME} Stage failed")
        logging.exception(e)
        raise e
