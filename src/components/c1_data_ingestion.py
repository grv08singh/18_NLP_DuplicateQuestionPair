import os
from src.logger import logging
from src.entities.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        logging.info("")