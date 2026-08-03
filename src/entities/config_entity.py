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

@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    preprocessed_data: Path
    y_path: Path
    q1_encoded: Path
    q2_encoded: Path
    base_emb_model: Path
    trained_emb_model: Path
    word_pair_emb: Path
    emb_matrix: Path
    q1_emb: Path
    q2_emb: Path
    processed_data: Path
    token_window: int
    emb_dim: int
    neg_samples: int
    epochs: int
    min_token_count: int
    max_words: int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model: Path
    val_report: Path

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    test_dir: Path
    evaluation_report: Path
    mlflow_uri: str