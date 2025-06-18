import argparse
import pickle

import einops
import torch

from generative_models.datasets import CircleDataset
from generative_models.evaluators import DDIMSampler, DDPMSampler
from generative_models.models import (
    MLPConfig,
    PointWiseMLPDiffusion,
    PointWiseMLPDiffusionConfig,
)
from utils import checkpoints_dir, data_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIFFUSION_MODES = {"ddpm": DDPMSampler, "ddim": DDIMSampler}


def inference(diffusion_mode):
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
        t_steps=50,
    )

    model = PointWiseMLPDiffusion(
        config=model_config,
    ).to(device)

    model.load_state_dict(state_dict)

    train_data = CircleDataset(data_dir("circle_dataset.pkl"))
    num_points = train_data.num_points_per_circle

    t_steps = 50
    sampler = DIFFUSION_MODES[diffusion_mode](model, t_steps, device)

    x_t, x_ts = sampler.sample(1, num_points)
    x_ts = einops.rearrange(x_ts, "t 1 n d -> t n d")
    x_steps = x_ts.detach().cpu().numpy()
    with open(data_dir("x_steps.pkl"), "wb") as f:
        pickle.dump(x_steps, f)
    print(f"Saved x_steps to {data_dir('x_steps.pkl')}")


def main():
    parser = argparse.ArgumentParser(
        prog="Inference", description="run inference on trained models"
    )
    parser.add_argument(
        "-d",
        "--diffusion_mode",
        default="ddpm",
        help=f"One of {list(DIFFUSION_MODES.keys())}",
    )
    args = parser.parse_args()

    assert args.diffusion_mode in DIFFUSION_MODES.keys(), (
        f"Diffusion mode {args.diffusion_mode} not in {list(DIFFUSION_MODES.keys())}"
    )

    inference(args.diffusion_mode)


if __name__ == "__main__":
    main()
