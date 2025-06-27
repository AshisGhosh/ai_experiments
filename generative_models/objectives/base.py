from abc import ABC, abstractmethod


class BaseObjective(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @abstractmethod
    def forward(self):
        pass
