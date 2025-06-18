from pathlib import Path

from generative_models.datasets import BaseDataset


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
