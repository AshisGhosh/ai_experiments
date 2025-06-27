import einops
import torch
import torch.nn as nn

from generative_models.schedulers import cosine_beta_schedule

from .base import BaseObjective


class DiffusionObjective(BaseObjective):
    def __init__(self, model, device, t_steps):
        super().__init__(model, device)
        self.t_steps = t_steps
        self.beta = cosine_beta_schedule(t_steps, s=0.008)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(device)
        self.criterion = nn.MSELoss().to(device)

    def forward(
        self,
        x,
    ):
        x = x.to(self.device)

        # create (B, 1) tensor of t_index
        t = (torch.rand(x.shape[0], device=self.device) * (self.t_steps - 1)).long()
        t = einops.repeat(t, "b -> b n 1", n=x.shape[1])
        # shape: (B, n, 1)
        a_bar = self.alpha_cumprod[t]

        added_noise = torch.randn_like(x, device=self.device)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * added_noise

        noise_guess = self.model(x_t, t)
        loss = self.criterion(noise_guess, added_noise)
        return noise_guess, loss
