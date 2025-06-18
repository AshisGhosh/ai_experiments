from dataclasses import dataclass
from pathlib import Path

import einops
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from generative_models.datasets import CircleDataset
from generative_models.evaluators import DDPMSampler, Evaluator
from generative_models.models import (
    MLPConfig,
    PointWiseMLPDiffusion,
    PointWiseMLPDiffusionConfig,
)
from generative_models.objectives import DiffusionObjective, circularity_metric
from generative_models.schedulers import cosine_beta_schedule
from utils import checkpoints_dir, data_dir, determinism_over_performance, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    model_path: Path = checkpoints_dir("ddpm_model.pth")
    num_epochs: int = 2000
    batch_size: int = 128
    learning_rate: float = 3e-4
    t_steps: int = 100
    eval_every: int = 100


class Trainer:
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
        self.objective = DiffusionObjective(
            model=self.model,
            criterion=self.criterion,
            device=device,
            t_steps=config.t_steps,
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

        self.sampler = DDPMSampler(self.model, config.t_steps, device)
        self.evaluator = Evaluator(self.sampler, {"circularity": circularity_metric})

    def train(self):
        for epoch in tqdm.trange(self.config.num_epochs, desc=f"Training"):
            epoch_loss = 0
            for batch in tqdm.tqdm(
                self.train_loader, desc=f"Epoch {epoch}", disable=True
            ):
                output, loss = self.objective.forward(batch)
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
            eval_batch_size = 64
            num_points = self.train_data.num_points_per_circle
            metrics = self.evaluator.evaluate(eval_batch_size, num_points=num_points)
            print(metrics)
        finally:
            self.model.train()


def main():
    set_seed(42)
    determinism_over_performance()
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
