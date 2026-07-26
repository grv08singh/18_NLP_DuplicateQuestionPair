import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logging
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    try:
        with open(path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file loaded successfully: {path}")
            return ConfigBox(content)
    except BoxValueError as e:
        logging.info(f"YAML file is empty: {path}")
        raise ValueError("YAML file is empty")
    except Exception as e:
        logging.info(f"Error occurred while loading YAML: {path}")
        raise e:

@ensure_annotations
def create_directories(path_list: list, verbose=True):
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Directory created: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"JSON file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        content = json.load(f)
    logging.info(f"JSON loaded: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logging.info(f"Binary file saved at {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(filename=path)
    logging.info(f"Binary file loaded from {path}")
    return data