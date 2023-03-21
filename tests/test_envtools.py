import pytest
import torch
from src import envtools


def test_packing():
    g = torch.tensor([
        [
            [0., 1., 2.],
            [-1., 0.5, -1.5],
            [1.2, 0, -0.5]
        ],
        [
            [-3., -2, -1.,],
            [0.5, 0.5, 0.0],
            [3., 2., 1.]
        ]
    ])
    x = torch.tensor([
        [
            [-1., 1., -1]
        ],
        [
            [2., 0., 2]
        ]
    ])
    output = envtools.pack_states(g, x, 2)
    ans = torch.tensor([
        [
            [3., 2., 1.],
            [0., 1., 2.],
            [-1., 0.5, -1.5],
            [1.2, 0, -0.5],
            [-1., 1., -1],
            [0., 0., 0.]
        ],
        [
            [3., 2., 1.],
            [-3., -2, -1.,],
            [0.5, 0.5, 0.0],
            [3., 2., 1.],
            [2., 0., 2],
            [0., 0., 0.]
        ]
    ])
    assert torch.allclose(output, ans)


@pytest.mark.parametrize("b,K,d", [(1, 3, 10), (8, 5, 64)])
@pytest.mark.parametrize("T,t", [(3, 0), (3, 1), (3, 3), (5, 4)])
def test_pack_unpack(b: int, K: int, T: int, t: int, d: int):
    g = torch.randn((b, K, d))
    x = torch.randn((b, t, d)) if t > 0 else None
    packed = envtools.pack_states(g, x, T)
    g_out, x_out = envtools.unpack_states(packed)
    assert (torch.allclose(g, g_out)
            and ((t == 0 and x_out is None) or torch.allclose(x, x_out)))
