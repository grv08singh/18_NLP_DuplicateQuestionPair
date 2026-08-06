import os
from src.logger import logging
from src.constants import *
from src.utils.common import read_yaml, create_directories, save_json
from src.entities.config_entity import DataIngestionConfig, DataPreProcessingConfig, DataProcessingConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        logging.info("Initialization started for class ConfigurationManager")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        logging.info("Initialization completed for class ConfigurationManager")
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logging.info("Entered Method: get_data_ingestion_config")
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            raw_data_file=config.raw_data_file,
            src_chatwords_file=config.src_chatwords_file,
            dest_chatwords_file=config.dest_chatwords_file
        )
        logging.info("Exited Method: get_data_ingestion_config")
        return data_ingestion_config
        
    def get_data_preprocessing_config(self) -> DataPreProcessingConfig:
        logging.info("Entered Method: get_data_preprocessing_config")
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        data_preprocessing_config = DataPreProcessingConfig(
            raw_data_file=config.raw_data_file,
            root_dir=config.root_dir,
            chatwords_file=config.chatwords_file,
            preprocessed_data=config.preprocessed_data,
            X_path=config.y_path,
            y_path=config.y_path,
        )
        logging.info("Exited Method: get_data_preprocessing_config")
        return data_preprocessing_config
    
    def get_data_processing_config(self) -> DataProcessingConfig:
        logging.info("Entered Method: get_data_processing_config")
        config = self.config.data_processing
        params = self.params
        create_directories([config.root_dir])
        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            preprocessed_data=config.preprocessed_data,
            X_path=config.X_path,
            q1_encoded=config.q1_encoded,
            q2_encoded=config.q2_encoded,
            base_emb_model=config.base_emb_model,
            trained_emb_model=config.trained_emb_model,
            word_pair_emb=config.word_pair_emb,
            emb_matrix=config.emb_matrix,
            q1_emb=config.q1_emb,
            q2_emb=config.q2_emb,
            emb_min_token_count=params.EMB_MIN_TOKEN_COUNT,
            emb_max_words=params.EMB_MAX_WORDS,
            emb_window_size=params.EMB_WINDOW_SIZE,
            emb_dim=params.EMB_DIM,
            emb_neg_samples=params.EMB_NEG_SAMPLES,
            emb_batch_size=params.EMB_BATCH_SIZE,
            emb_epochs=params.EMB_EPOCHS
        )
        logging.info("Exited Method: get_data_processing_config")
        return data_processing_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        logging.info("Entered Method: get_prepare_base_model_config")
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model=config.base_model
        )
        logging.info("Exited Method: get_prepare_base_model_config")
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        logging.info("Entered Method: get_training_config")
        config = self.config.training
        create_directories([config.root_dir])
        training_config = TrainingConfig(
            root_dir=config.root_dir,
            trained_model=config.trained_model,
            val_report=config.val_report
        )
        logging.info("Exited Method: get_training_config")
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        logging.info("Entered Method: get_evaluation_config")
        config = self.config.evaluation
        create_directories([config.root_dir])
        evaluation_config = EvaluationConfig(
            root_dir=config.root_dir,
            test_dir=config.test_dir,
            evaluation_report=config.evaluation_report,
            mlflow_uri=config.mlflow_uri
        )
        logging.info("Exited Method: get_evaluation_config")
        print("1234")
        return evaluation_config