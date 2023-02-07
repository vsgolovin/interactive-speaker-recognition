import torch
from torch import nn, Tensor
from torch.nn import functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        e = self.mlp(torch.cat([q, k], dim=-1))
        alpha = F.softmax(e, dim=-2)
        v_hat = torch.sum(alpha * v, dim=-2)
        return v_hat
