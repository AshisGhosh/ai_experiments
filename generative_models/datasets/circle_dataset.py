from pathlib import Path

import einops
import numpy as np

from generative_models.datasets import BaseDataset
from utils import data_dir


class CircleDataset(BaseDataset):
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self._num_points_per_circle = None

    @property
    def num_points_per_circle(self):
        if self._num_points_per_circle is None:
            self._num_points_per_circle = self.data.shape[1]
        return self._num_points_per_circle

    def __getitem__(self, idx):
        return self.data[idx]


class OrderedCircleDataset(CircleDataset):
    def __init__(self, data_path: Path):
        super().__init__(data_path)

    @property
    def data(self):
        if self._data is None:
            data = super().data
            # sort by polar angle for each circle using vectorized operations
            polar_angles = np.arctan2(data[..., 1], data[..., 0])  # shape: [1000, 50]
            sorted_indices = np.argsort(polar_angles, axis=1)  # shape: [1000, 50]

            batch_indices = einops.repeat(
                np.arange(data.shape[0]), "b -> b n", n=data.shape[1]
            )
            self._data = data[batch_indices, sorted_indices]
        return self._data


if __name__ == "__main__":
    dataset = OrderedCircleDataset(data_dir("circle_dataset.pkl"))
    print(dataset.data.shape)
