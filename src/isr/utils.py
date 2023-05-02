import torch
from torch import Tensor


def pairwise_l2_distances(x1: Tensor, x2: Tensor) -> Tensor:
    assert x1.ndim == 2 and x2.ndim == 2 and x1.size(1) == x2.size(1)
    x1_stack = torch.stack([x1] * x2.size(0), dim=1)
    x2_stack = torch.stack([x2] * x1.size(0), dim=0)
    return ((x1_stack - x2_stack)**2).sum(2).sqrt()


def pairwise_mse(x1: Tensor, x2: Tensor) -> Tensor:
    assert x1.ndim == 2 and x2.ndim == 2 and x1.size(1) == x2.size(1)
    x1_stack = torch.stack([x1] * x2.size(0), dim=1)
    x2_stack = torch.stack([x2] * x1.size(0), dim=0)
    return ((x1_stack - x2_stack)**2).mean(2)
