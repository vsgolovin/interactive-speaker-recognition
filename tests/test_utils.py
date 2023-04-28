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
