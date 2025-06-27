import torch
import torch.nn as nn

from .base import BaseObjective


class SequenceObjective(BaseObjective):
    def __init__(self, model, device):
        super().__init__(model, device)
        self.criterion = nn.MSELoss().to(device)

    def forward(self, x):
        inputs = x[:, :-1, :].to(self.device)
        targets = x[:, 1:, :].to(self.device)

        outputs = self.model(inputs)
        loss = self.get_loss(outputs, targets)
        return outputs, loss

    def get_loss(self, outputs, targets):
        # convert x, y to polar coordinates
        target_radius = torch.sqrt(targets[:, :, 0] ** 2 + targets[:, :, 1] ** 2)
        target_angle = torch.atan2(targets[:, :, 1], targets[:, :, 0])

        output_radius = torch.sqrt(outputs[:, :, 0] ** 2 + outputs[:, :, 1] ** 2)
        output_angle = torch.atan2(outputs[:, :, 1], outputs[:, :, 0])

        radius_loss = self.criterion(output_radius, target_radius)
        angle_loss = self.criterion(output_angle, target_angle)

        loss = radius_loss + angle_loss
        return loss
