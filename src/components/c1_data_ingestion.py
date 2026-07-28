import os
from src.logger import logging
from src.entities.config_entity import DataIngestionConfig
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        logging.info("DataIngestion class Initialization started")
        self.config = config
        logging.info("DataIngestion class Initialization completed")
    
    def create_raw_data(self):
        logging.info("Entered Method create_raw_data of DataIngestion")
        src_path = self.config.local_data_file
        dest_path = self.config.raw_data_file
        shutil.copy(src_path, dest_path)
        logging.info("Exited Method create_raw_data of DataIngestion")