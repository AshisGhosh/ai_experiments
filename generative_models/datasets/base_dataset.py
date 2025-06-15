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
