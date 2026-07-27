from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    raw_data_file: Path

@dataclass(frozen=True)
class DataPreProcessingConfig:
    root_dir: Path
    preprocessed_data: Path

@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    processed_data: Path

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