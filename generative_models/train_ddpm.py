from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from generative_models.datasets import CircleDataset
from generative_models.models import (
    MLPConfig,
    PointWiseMLPDiffusion,
    PointWiseMLPDiffusionConfig,
)
from generative_models.objectives import determine_circularity
from generative_models.schedulers import cosine_beta_schedule
from utils import checkpoints_dir, data_dir, determinism_over_performance, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    model_path: str = checkpoints_dir("ddpm_model.pth")
    num_epochs: int = 2000
    batch_size: int = 128
    learning_rate: float = 3e-4
    t_steps: int = 100
    eval_every: int = 100


class DDPMTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_data = CircleDataset(data_dir("circle_dataset.pkl"))
        self.train_loader = DataLoader(
            self.train_data, batch_size=config.batch_size, shuffle=True
        )
        self.time_dim = 1
        self.input_dim = self.train_data[0].shape[-1] + self.time_dim
        self.output_dim = self.train_data[0].shape[-1]

        self.mlp_config = MLPConfig(
            input_dim=self.input_dim,
            hidden_layers=[256, 256, 256],
            output_dim=self.output_dim,
            activation="relu",
        )
        self.model_config = PointWiseMLPDiffusionConfig(
            mlp_config=self.mlp_config,
            t_steps=config.t_steps,
        )
        self.model = PointWiseMLPDiffusion(config=self.model_config).to(device)
        print(f"Model config: {self.model_config}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.MSELoss().to(device)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

    def train(self):
        t_steps = self.model_config.t_steps
        beta = cosine_beta_schedule(t_steps, s=0.008)
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

        for epoch in tqdm.trange(self.config.num_epochs, desc=f"Training"):
            epoch_loss = 0
            for batch in tqdm.tqdm(
                self.train_loader, desc=f"Epoch {epoch}", disable=True
            ):
                x = batch.to(device)

                # create (B, 1) tensor of t_index
                t = (torch.rand(x.shape[0], device=device) * (t_steps - 1)).long()
                t = einops.repeat(t, "b -> b n 1", n=x.shape[1])
                # shape: (B, n, 1)
                a_bar = alpha_cumprod[t]

                added_noise = torch.randn_like(x, device=device)
                x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * added_noise

                noise_guess = self.model(x_t, t)

                loss = self.criterion(noise_guess, added_noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            if epoch % 100 == 0:
                tqdm.tqdm.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            if epoch % self.config.eval_every == 0:
                self.evaluate()

        torch.save(self.model.state_dict(), self.config.model_path)

    def evaluate(self):
        self.model.eval()
        try:
            num_points = self.train_data.num_points_per_circle

            batch_size = 64
            x_t = torch.randn(batch_size, num_points, 2, device=device)

            t_steps = 25
            beta = cosine_beta_schedule(t_steps, s=0.008).to(device)
            alpha = 1 - beta
            alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

            for t_idx in tqdm.tqdm(range(t_steps, 0, -1)):
                t = torch.tensor([t_idx - 1], device=device)
                t = einops.repeat(t, "1 -> b n 1", b=x_t.shape[0], n=x_t.shape[1])

                # add batch dimension
                noise_t = self.model(x_t, t)
                x_t_minus_1 = (
                    1
                    / torch.sqrt(alpha[t_idx - 1])
                    * (
                        x_t
                        - (1 - alpha[t_idx - 1])
                        / torch.sqrt(1 - alpha_cumprod[t_idx - 1])
                        * noise_t
                    )
                )
                if t_idx > 1:
                    x_t_minus_1 = x_t_minus_1 + torch.randn_like(
                        x_t_minus_1
                    ) * torch.sqrt(beta[t_idx - 1])
                x_t = x_t_minus_1

            circularity = determine_circularity(x_t)
            print(f"Circularity: {circularity}")
            return circularity
        finally:
            self.model.train()


def main():
    set_seed(42)
    determinism_over_performance()
    config = TrainingConfig()
    trainer = DDPMTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
