from abc import ABC, abstractmethod


class BaseObjective(ABC):
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    @abstractmethod
    def forward(self):
        pass
