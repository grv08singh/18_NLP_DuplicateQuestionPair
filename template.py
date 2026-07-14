import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')

project_name = "Duplication_Question_Recognition"

list_of_files = [
    "app.py",
    ".github/workflows/.gitkeep",
    "artifacts/.gitkeep",
    "logs/.gitkeep",
    "notebooks/data/.gitkeep",
    "notebooks/01_DataIngestion.ipynb",
    "notebooks/02_DataPreprocessing.ipynb",
    "notebooks/03_EmbeddingsGeneration.ipynb",
    "notebooks/04_BaseModelPreparation.ipynb",
    "notebooks/05_ModelTraining.ipynb",
    "notebooks/06_ModelEvalutation.ipynb",
    "notebooks/07_Prediction.ipynb",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_preprocessing.py",
    "src/components/embeddings_generation.py",
    "src/components/base_model_preparation.py",
    "src/components/model_training.py",
    "src/components/model_evaluation.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/constants/__init__.py",
    "src/entities/__init__.py",
    "src/entities/config_entity.py",
    "src/pipelines/__init__.py",
    "src/pipelines/data_ingestion_pipeline.py",
    "src/pipelines/data_preprocessing_pipeline.py",
    "src/pipelines/embeddings_generation_pipeline.py",
    "src/pipelines/base_model_preparation_pipeline.py",
    "src/pipelines/model_training_pipeline.py",
    "src/pipelines/model_evaluation_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "templates/index.html",
    ".gitignore",
    "config.yaml",
    "dvc.yaml",
    "main.py",
    "params.yaml",
    "README.md",
    "requirements.txt",
    "setup.py",
    "template.py",
    "Dockerfile"
]

for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Directory created: {file_dir}")

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as f:
            pass
        logging.info(f"File created: {file_path}")
    else:
        logging.info(f"File already exists and is not empty: {file_path}")