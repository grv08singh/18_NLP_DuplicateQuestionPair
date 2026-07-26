import os
from src.constants import *
from src.utils.common import read_yaml, create_directories, save_json
from src.entities.config_entity import DataIngestionConfig
import logging

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        logging.info(f"ConfigurationManager class initialization started")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        logging.info("ConfigurationManager class initialization completed")
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logging.info("Entered Method: get_data_ingestion_config")
        config = self.config.data_ingestion
        create_directories([config.root_dir])