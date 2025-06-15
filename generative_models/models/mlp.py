from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_layers: list[int]
    activation: str = "relu"


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        layers = []
        layers.append(nn.Linear(config.input_dim, config.hidden_layers[0]))
        for i in range(len(config.hidden_layers) - 1):
            layers.append(
                nn.Linear(config.hidden_layers[i], config.hidden_layers[i + 1])
            )
            if config.activation == "relu":
                layers.append(nn.ReLU())
        layers.append(nn.Linear(config.hidden_layers[-1], config.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
