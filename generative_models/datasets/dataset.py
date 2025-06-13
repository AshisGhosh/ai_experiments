import pickle

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._data = None

    @property
    def data(self):
        if self._data is None:
            with open(self.data_path, "rb") as f:
                self._data = pickle.load(f)
        return self._data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CircleDataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self._num_points_per_circle = None

    @property
    def num_points_per_circle(self):
        if self._num_points_per_circle is None:
            self._num_points_per_circle = self.data.shape[1]
        return self._num_points_per_circle

    def __getitem__(self, idx):
        return self.data[idx]
