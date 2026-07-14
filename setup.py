from setuptools import setup, find_packages
from typing import List
import os

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    "This function will return the list of requirements"
    try:
        with open(file_path) as f:
            requirements = [r.strip() for r in f.readlines()]
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
            return requirements
    except FileNotFoundError:
        print(f"The file {file_path} doesn't exist.")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

SRC_NAME = "NLP_Duplicate_Question_Pair"
VERSION = "0.0.1"
AUTHOR = "Gaurav Singh"
AUTHOR_EMAIL = "grv08singh@gmail.com"
AUTHOR_USERNAME = "grv08singh"
REPO_NAME = "18_NLP_DuplicateQuestionPair"
setup(
    name=SRC_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}"
)

get_requirements('requirements.txt')