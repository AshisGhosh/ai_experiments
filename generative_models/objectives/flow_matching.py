import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_models.objectives.base import BaseObjective
from generative_models.schedulers import cosine_beta_schedule


class FlowMatchingObjective(BaseObjective):
    def __init__(self, model, device, t_steps):
        super().__init__(model, device)
        self.t_steps = t_steps
        self.beta = cosine_beta_schedule(t_steps, s=0.008)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(device)
        self.criterion = nn.MSELoss().to(device)

    def forward(self, x):
        x = x.to(self.device)

        t = torch.rand(x.shape[0], device=self.device) * (self.t_steps - 1)
        t = t.long()
        t = einops.repeat(t, "b -> b n 1", n=x.shape[1])
        a_bar = self.alpha_cumprod[t]

        added_noise = torch.randn_like(x, device=self.device)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * added_noise

        dt = 1
        t_prime = t + dt
        a_bar_prime = self.alpha_cumprod[t_prime]

        x_tp = torch.sqrt(a_bar_prime) * x + torch.sqrt(1 - a_bar_prime) * added_noise
        v_target = (x_tp - x_t) / dt
        v_pred = self.model(x_t, t)

        loss = self.criterion(v_pred, v_target)
        return x_t, loss
