from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    raw_data_file: Path
    src_chatwords_file: Path
    dest_chatwords_file: Path

@dataclass(frozen=True)
class DataPreProcessingConfig:
    root_dir: Path
    raw_data_file: Path
    chatwords_file: Path
    preprocessed_data: Path
    X_path: Path
    y_path: Path

@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    model_dir: Path
    X_path: Path
    word2idx_path: Path
    trained_emb_model: Path
    bert_emb_model: Path
    word_pair_emb: Path
    emb_matrix: Path
    emb_data_own: Path
    q1_emb_own: Path
    q2_emb_own: Path
    emb_data_bert: Path
    q1_emb_bert: Path
    q2_emb_bert: Path
    emb_min_token_count: int
    emb_max_words: int
    emb_window_size: int
    emb_dim: int
    emb_neg_samples: int
    emb_batch_size: int
    emb_epochs: int
    own_emb_model: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_dir: Path
    y_path: Path
    emb_matrix: Path
    emb_data_own: Path
    q1_emb_own: Path
    q2_emb_own: Path
    emb_data_bert: Path
    q1_emb_bert: Path
    q2_emb_bert: Path
    trained_model_with_own_emb: Path
    trained_model_with_bert_emb: Path
    own_val_report: Path
    bert_val_report: Path
    own_emb_model: bool
    epochs: int

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    test_dir: Path
    evaluation_report: Path
    mlflow_uri: str

@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    chatwords_file: Path
    word2idx_path: Path
    emb_matrix: Path
    own_emb_model: bool
    bert_emb_model: Path
    trained_model_with_own_emb: Path
    trained_model_with_bert_emb: Path
    seq_len: int