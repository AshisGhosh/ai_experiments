import pickle

import einops
import matplotlib.pyplot as plt
import torch
import tqdm

from generative_models.datasets import CircleDataset
from generative_models.models import (
    MLPConfig,
    PointWiseMLPDiffusion,
    PointWiseMLPDiffusionConfig,
)
from generative_models.schedulers import cosine_beta_schedule
from generative_models.train_ddpm import TrainingConfig
from utils import checkpoints_dir, data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference():
    state_dict = torch.load(checkpoints_dir("ddpm_model.pth"), weights_only=True)

    # get dimensions from state dict
    state_dict_keys = list(state_dict.keys())
    first_layer = state_dict_keys[0]
    last_layer = state_dict_keys[-1]
    input_dim = state_dict[first_layer].shape[1]
    output_dim = state_dict[last_layer].shape[0]

    mlp_config = MLPConfig(
        input_dim=input_dim,
        hidden_layers=[256, 256, 256],
        output_dim=output_dim,
        activation="relu",
    )
    model_config = PointWiseMLPDiffusionConfig(
        mlp_config=mlp_config,
        t_steps=25,
    )

    model = PointWiseMLPDiffusion(
        config=model_config,
    ).to(device)

    model.load_state_dict(state_dict)

    train_data = CircleDataset(data_dir("circle_dataset.pkl"))
    num_points = train_data.num_points_per_circle

    initial_noise = torch.randn(num_points, 2, device=device)
    x_steps = [initial_noise]
    x_t = einops.rearrange(initial_noise, "n d -> 1 n d").to(device)

    t_steps = 25
    beta = cosine_beta_schedule(t_steps, s=0.008).to(device)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    for t_idx in tqdm.tqdm(range(t_steps, 0, -1)):
        t = torch.tensor([t_idx - 1], device=device)
        t = einops.repeat(t, "1 -> 1 n 1", n=x_t.shape[1])

        # add batch dimension
        noise_t = model(x_t, t)
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
            x_t_minus_1 = x_t_minus_1 + torch.randn_like(x_t_minus_1) * torch.sqrt(
                beta[t_idx - 1]
            )
        x_t = x_t_minus_1
        x_steps.append(einops.rearrange(x_t, "1 n d -> n d"))

    x_steps = torch.stack(x_steps, dim=0).detach().cpu().numpy()
    with open(data_dir("x_steps.pkl"), "wb") as f:
        pickle.dump(x_steps, f)
    print(f"Saved x_steps to {data_dir('x_steps.pkl')}")


def main():
    inference()


if __name__ == "__main__":
    main()
