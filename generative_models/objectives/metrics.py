import einops
import torch


def circularity_metric(points: torch.Tensor) -> torch.Tensor:
    """
    Determine how well the points fit a circle.
    """
    center = torch.mean(points, dim=1)
    center = einops.rearrange(center, "b d -> b 1 d")
    radii = torch.linalg.norm(points - center, dim=2)
    mean_radius = torch.mean(radii)
    std_radius = torch.std(radii)
    return std_radius / mean_radius
