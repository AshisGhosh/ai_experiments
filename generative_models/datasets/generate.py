import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import data_dir


@dataclass
class GenerateCircleDatasetConfig:
    num_data_points: int
    num_points_per_circle: int
    radius: float
    center: tuple[float, float]
    noise_level: float
    output_path: str = data_dir("circle_dataset.pkl")
    pickle_protocol: int = 4
    device: str = "cpu"


def generate_circle_dataset(config: GenerateCircleDatasetConfig):
    """
    Generate a dataset of circles.
    """
    circle_points = (
        torch.rand(
            config.num_data_points, config.num_points_per_circle, device=config.device
        )
        * 2
        * np.pi
    )
    x = config.radius * torch.cos(circle_points) + config.center[0]
    y = config.radius * torch.sin(circle_points) + config.center[1]
    points = torch.stack([x, y], dim=1)
    noise = torch.randn(points.shape, device=config.device) * config.noise_level
    points += noise

    output_path = os.path.join(os.path.dirname(__file__), config.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(points, f, protocol=config.pickle_protocol)

    logger.info(f"Saved dataset {config} to {config.output_path}")


if __name__ == "__main__":
    config = GenerateCircleDatasetConfig(
        num_data_points=1000,
        num_points_per_circle=50,
        radius=1.0,
        center=(0.0, 0.0),
        noise_level=0.01,
    )
    generate_circle_dataset(config)
