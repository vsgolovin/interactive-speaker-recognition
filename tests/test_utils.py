import pytest
import torch
from isr import utils


def test_l2_distances():
    x = torch.tensor([
        [1., 2., 3.],
        [0., 0., 0.],
        [1., 0., 1.]
    ])
    y = torch.tensor([
        [1., 1., 1.],
        [-2, 0., 1.],
    ])
    out = utils.pairwise_l2_distances(x, y)
    ans = torch.tensor([
        [2.236, 4.123],
        [1.732, 2.236],
        [1.000, 3.000]
    ])
    assert torch.allclose(out, ans, atol=5e-4)


@pytest.mark.parametrize("d", (8, 32, 512))
@pytest.mark.parametrize("n1,n2", [(5, 5), (10, 2), (1, 16)])
def test_pairwise_mse(n1: int, n2: int, d: int):
    x1 = torch.randn((n1, d))
    x2 = torch.randn((n2, d))
    ans = torch.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            ans[i, j] = ((x1[i] - x2[j])**2).mean()
    out = utils.pairwise_mse(x1, x2)
    assert torch.allclose(out, ans)
