from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def generative_models_dir(dir_name: Optional[str] = None) -> Path:
    generative_models_dir = get_project_root() / "generative_models"
    if dir_name is not None:
        generative_models_dir = generative_models_dir / dir_name
    return generative_models_dir


def data_dir(dir_name: Optional[str] = None) -> Path:
    data_dir = generative_models_dir("datasets/data")
    if dir_name is not None:
        data_dir = data_dir / dir_name
    return data_dir


def checkpoints_dir(dir_name: Optional[str] = None) -> Path:
    checkpoints_dir = generative_models_dir("checkpoints")
    if dir_name is not None:
        checkpoints_dir = checkpoints_dir / dir_name
    return checkpoints_dir
