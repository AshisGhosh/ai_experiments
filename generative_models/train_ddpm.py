from dataclasses import dataclass

import einops
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from generative_models.datasets import CircleDataset
from generative_models.models import MLP, MLPConfig
from generative_models.schedulers import cosine_beta_schedule
from utils import data_dir, determinism_over_performance, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    model_path: str = "ddpm_model.pth"
    num_epochs: int = 5000
    batch_size: int = 128
    learning_rate: float = 3e-4
    t_steps: int = 100


def train(
    config: TrainingConfig,
):
    train_data = CircleDataset(data_dir("circle_dataset.pkl"))
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    xypoints_dim = train_data[0].shape[-1]
    time_dim = 1
    input_dim = xypoints_dim + time_dim
    output_dim = xypoints_dim

    model_config = MLPConfig(
        input_dim=input_dim,
        hidden_layers=[512, 512, 512],
        output_dim=output_dim,
        activation="relu",
    )

    model = MLP(
        config=model_config,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss().to(device)

    t_steps = config.t_steps
    beta = cosine_beta_schedule(t_steps, s=0.008)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    for epoch in tqdm.trange(config.num_epochs, desc="Training"):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", disable=True):
            x = batch.to(device)

            # create (B, 1) tensor of t_index
            t = (torch.rand(x.shape[0], device=device) * (t_steps - 1)).long()
            t = einops.rearrange(t, "b -> b 1")
            # shape: (B, 1)
            a_bar = alpha_cumprod[t].view(-1, 1)

            added_noise = torch.randn_like(x, device=device)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * added_noise

            normalized_t = t.float() / (t_steps - 1)

            model_inputs = torch.cat([x_t, normalized_t], dim=-1)
            noise_guess = model(model_inputs)

            loss = criterion(noise_guess, added_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        tqdm.tqdm.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.model_path)


def main():
    set_seed(42)
    determinism_over_performance()
    config = TrainingConfig()
    train(config)


if __name__ == "__main__":
    main()
