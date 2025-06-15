import einops

from generative_models.datasets import BaseDataset


class CircleDataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self._num_points_per_circle = None

    @property
    def num_points_per_circle(self):
        if self._num_points_per_circle is None:
            self._num_points_per_circle = self.data.shape[-1] // 2
        return self._num_points_per_circle

    def __getitem__(self, idx):
        data = self.data[idx]
        # transfrom from [X], [Y] to [x1, y1, x2, y2, ...]
        return einops.rearrange(data, "XY n -> (n XY)")
