import os
from src.logger import logging
from src.constants import *
from src.utils.common import read_yaml, create_directories, save_json
from src.entities.config_entity import DataIngestionConfig, DataPreProcessingConfig, DataProcessingConfig, TrainingConfig, EvaluationConfig, PredictionConfig

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
            X_path=config.X_path,
            y_path=config.y_path,
        )
        logging.info("Exited Method: get_data_preprocessing_config")
        return data_preprocessing_config
    
    def get_data_processing_config(self) -> DataProcessingConfig:
        logging.info("Entered Method: get_data_processing_config")
        config = self.config.data_processing
        params = self.params
        create_directories([config.root_dir, config.model_dir])
        data_processing_config = DataProcessingConfig(
            root_dir=config.root_dir,
            model_dir=config.model_dir,
            X_path=config.X_path,
            word2idx_path=config.word2idx_path,
            trained_emb_model=config.trained_emb_model,
            bert_emb_model=config.bert_emb_model,
            word_pair_emb=config.word_pair_emb,
            emb_matrix=config.emb_matrix,
            emb_data_own=config.emb_data_own,
            q1_emb_own=config.q1_emb_own,
            q2_emb_own=config.q2_emb_own,
            emb_data_bert=config.emb_data_bert,
            q1_emb_bert=config.q1_emb_bert,
            q2_emb_bert=config.q2_emb_bert,
            emb_min_token_count=params.EMB_MIN_TOKEN_COUNT,
            emb_max_words=params.EMB_MAX_WORDS,
            emb_window_size=params.EMB_WINDOW_SIZE,
            emb_dim=params.EMB_DIM,
            emb_neg_samples=params.EMB_NEG_SAMPLES,
            emb_batch_size=params.EMB_BATCH_SIZE,
            emb_epochs=params.EMB_EPOCHS,
            own_emb_model=params.OWN_EMB_MODEL
        )
        logging.info("Exited Method: get_data_processing_config")
        return data_processing_config
    
    def get_training_config(self) -> TrainingConfig:
        logging.info("Entered Method: get_training_config")
        config = self.config.training
        params = self.params
        create_directories([config.root_dir])
        training_config = TrainingConfig(
            root_dir=config.root_dir,
            model_dir=config.model_dir,
            y_path=config.y_path,
            emb_matrix=config.emb_matrix,
            emb_data_own=config.emb_data_own,
            q1_emb_own=config.q1_emb_own,
            q2_emb_own=config.q2_emb_own,
            emb_data_bert=config.emb_data_bert,
            q1_emb_bert=config.q1_emb_bert,
            q2_emb_bert=config.q2_emb_bert,
            trained_model_with_own_emb=config.trained_model_with_own_emb,
            trained_model_with_bert_emb=config.trained_model_with_bert_emb,
            own_val_report=config.own_val_report,
            bert_val_report=config.bert_val_report,
            own_emb_model=params.OWN_EMB_MODEL,
            epochs=params.EPOCHS
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
    
    def get_prediction_config(self) -> PredictionConfig:
        logging.info("Entered Method: get_prediction_config")
        config = self.config.prediction
        create_directories([config.root_dir])
        prediction_config = PredictionConfig(
            root_dir=config.root_dir,
            chatwords_file=config.chatwords_file,
            word2idx_path=config.word2idx_path,
            emb_matrix=config.emb_matrix,
            own_emb_model=self.params.OWN_EMB_MODEL,
            bert_emb_model=config.bert_emb_model,
            trained_model_with_own_emb=config.trained_model_with_own_emb,
            trained_model_with_bert_emb=config.trained_model_with_bert_emb,
            seq_len=self.params.SEQ_LEN
        )
        logging.info("Exited Method: get_prediction_config")
        return prediction_config