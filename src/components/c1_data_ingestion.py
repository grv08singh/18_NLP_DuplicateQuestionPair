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
        data_src = self.config.local_data_file
        data_dest = self.config.raw_data_file
        shutil.copy(data_src, data_dest)
        
        chatwords_src = self.config.src_chatwords_file
        chatwords_dest = self.config.dest_chatwords_file
        shutil.copy(chatwords_src, chatwords_dest)
        logging.info("Exited Method create_raw_data of DataIngestion")