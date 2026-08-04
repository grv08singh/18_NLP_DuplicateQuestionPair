import os
from box.exceptions import BoxValueError
import yaml
from src.logger import logging
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import json
import joblib

@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    logging.info("Entered Method read_yaml")
    try:
        with open(path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file loaded successfully: {path}")
            logging.info("Exited Method read_yaml")
            return ConfigBox(content)
    except BoxValueError as e:
        logging.info(f"YAML file is empty: {path}")
        raise ValueError("YAML file is empty")
    except Exception as e:
        logging.info(f"Error occurred while loading YAML: {path}")
        raise e

@ensure_annotations
def create_directories(path_list: list, verbose=True):
    logging.info("Entered Method create_directories")
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Directory created: {path}")
    logging.info("Exited Method create_directories")

@ensure_annotations
def save_json(path: Path, data: dict):
    logging.info("Entered Method save_json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"JSON file saved at {path}")
    logging.info("Exited Method save_json")

@ensure_annotations
def load_json(path: Path) -> dict:
    logging.info("Entered Method load_json")
    with open(path, 'r') as f:
        content = json.load(f)
    logging.info(f"JSON loaded: {path}")
    logging.info("Exited Method load_json")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    logging.info("Entered Method save_bin")
    joblib.dump(value=data, filename=path)
    logging.info(f"Binary file saved at {path}")
    logging.info("Exited Method save_bin")

@ensure_annotations
def load_bin(path: Path) -> Any:
    logging.info("Entered Method load_bin")
    data = joblib.load(filename=path)
    logging.info(f"Binary file loaded from {path}")
    logging.info("Exited Method load_bin")
    return data