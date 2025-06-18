from abc import ABC, abstractmethod

import einops
import torch
import tqdm

from generative_models.schedulers import cosine_beta_schedule


class DiffusionSampler:
    def __init__(self, model, t_steps, device):
        self.model = model
        self.t_steps = t_steps
        self.beta = cosine_beta_schedule(t_steps, s=0.008).to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(device)
        self.device = device

    @abstractmethod
    @torch.no_grad()
    def sample(self, batch_size, *args):
        pass


class DDPMSampler(DiffusionSampler):
    @torch.no_grad()
    def sample(self, batch_size, num_points):
        x_t = torch.randn(batch_size, num_points, 2, device=self.device)
        x_ts = [x_t]
        for t_idx in tqdm.tqdm(range(self.t_steps, 0, -1)):
            t = torch.tensor([t_idx - 1], device=self.device)
            t = einops.repeat(t, "1 -> b n 1", b=batch_size, n=num_points)

            # add batch dimension
            noise_t = self.model(x_t, t)
            x_t_minus_1 = (
                1
                / torch.sqrt(self.alpha[t_idx - 1])
                * (
                    x_t
                    - (1 - self.alpha[t_idx - 1])
                    / torch.sqrt(1 - self.alpha_cumprod[t_idx - 1])
                    * noise_t
                )
            )
            if t_idx > 1:
                x_t_minus_1 = x_t_minus_1 + torch.randn_like(x_t_minus_1) * torch.sqrt(
                    self.beta[t_idx - 1]
                )
            x_t = x_t_minus_1
            x_ts.append(x_t)
        x_ts = torch.stack(x_ts, dim=0)
        return x_t, x_ts


class DDIMSampler(DiffusionSampler):
    def __init__(self, model, t_steps, device):
        super().__init__(model, t_steps, device)
        self.alpha_cumprod = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_cumprod]
        )

    @torch.no_grad()
    def sample(self, batch_size, num_points):
        eta = 0.2
        x_t = torch.randn(batch_size, num_points, 2, device=self.device)
        x_ts = [x_t]
        for t_idx in tqdm.tqdm(range(self.t_steps, 0, -1)):
            t = torch.tensor([t_idx - 1], device=self.device)
            t = einops.repeat(t, "1 -> 1 n 1", n=x_t.shape[1])

            # add batch dimension
            noise_t = self.model(x_t, t)
            a_bar = self.alpha_cumprod[t_idx - 1]
            x_0_pred = (x_t - torch.sqrt(1 - a_bar) * noise_t) / torch.sqrt(a_bar)

            a_bar_t_minus_1 = self.alpha_cumprod[t_idx - 2]

            sigma_t = torch.sqrt((1 - a_bar_t_minus_1) / (1 - a_bar)) * eta
            sigma_t = torch.clamp(
                sigma_t, max=0.999
            )  # Prevent sigma_t from getting too large

            # Ensure the term under sqrt is positive
            noise_scale = torch.sqrt(
                torch.clamp(1 - a_bar_t_minus_1 - sigma_t**2, min=0.0)
            )
            mean_t = torch.sqrt(a_bar_t_minus_1) * x_0_pred + noise_scale * noise_t

            if eta > 0 and t_idx > 1:
                x_t_minus_1 = mean_t + torch.randn_like(x_t) * sigma_t
            else:
                x_t_minus_1 = mean_t

            x_t = x_t_minus_1
            x_ts.append(x_t)
        x_ts = torch.stack(x_ts, dim=0)
        return x_t, x_ts
