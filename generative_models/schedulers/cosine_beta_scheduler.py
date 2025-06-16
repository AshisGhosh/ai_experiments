import matplotlib.pyplot as plt
import numpy as np
import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.float()


def visualize_beta_schedule(timesteps=100, s=0.008):
    betas = cosine_beta_schedule(timesteps, s)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Plot beta values
    ax1.plot(betas.numpy(), label="β")
    ax1.set_title("Beta Schedule")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Beta Value")
    ax1.grid(True)

    # Plot cumulative noise levels
    ax2.plot(alphas_cumprod.numpy(), label="α_cumprod")
    ax2.set_title("Cumulative Noise Level")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Alpha")
    ax2.grid(True)

    # Plot noise level at each step
    ax3.plot((1 - alphas_cumprod).numpy(), label="Noise Level")
    ax3.set_title("Noise Level at Each Step")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Noise Level (1 - α_cumprod)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"beta_schedule_s={s}.png")
    plt.close()


if __name__ == "__main__":
    # Test different s values
    for s in [0.008, 0.01, 0.02, 0.05]:  # Back to original DDPM values
        print(f"\nVisualizing schedule with s={s}")
        visualize_beta_schedule(s=s)
