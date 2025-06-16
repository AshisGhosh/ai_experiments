from dataclasses import dataclass

import einops
import torch
import torch.nn as nn

from generative_models.models.mlp import MLP, MLPConfig


@dataclass
class PointWiseMLPDiffusionConfig:
    mlp_config: MLPConfig
    t_steps: int


class PointWiseMLPDiffusion(nn.Module):
    def __init__(self, config: PointWiseMLPDiffusionConfig):
        super().__init__()
        self.config = config
        self.mlp = MLP(config.mlp_config)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.float() / (self.config.t_steps - 1)
        x_t = torch.cat([x, t], dim=-1)
        return self.mlp(x_t)
