import pytest
import torch
from isr import envtools


def test_packing():
    g = torch.tensor([
        [
            [0., 1., 2.],
            [-1., 0.5, -1.5],
            [1.2, 0, -0.5]
        ],
        [
            [-3., -2, -1.],
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
            [-3., -2, -1.],
            [0.5, 0.5, 0.0],
            [3., 2., 1.],
            [2., 0., 2],
            [0., 0., 0.]
        ]
    ])
    assert torch.allclose(output, ans)


def test_packing_for_verifier():
    g = torch.tensor([
        [0., 1., 2.],
        [-1., 0.5, -1.5],
        [1.2, 0, -0.5]
    ])
    x = torch.tensor([
        [
            [-1., 1., -1],
            [-3., 1., 2.]
        ],
        [
            [2., 0., 2],
            [.4, 1., -.9]
        ],
        [
            [1., 0.5, 1.],
            [1.1, -.2, -.5]
        ]
    ])
    output = envtools.pack_states(g, x, 3)
    ans = torch.tensor([
        [
            [1., 3., 2.],
            [0., 1., 2.],
            [-1., 1., -1],
            [-3., 1., 2.],
            [0., 0., 0.]
        ],
        [
            [1., 3., 2.],
            [-1., 0.5, -1.5],
            [2., 0., 2],
            [.4, 1., -.9],
            [0., 0., 0.]
        ],
        [
            [1., 3., 2.],
            [1.2, 0, -0.5],
            [1., 0.5, 1.],
            [1.1, -.2, -.5],
            [0., 0., 0.]
        ]
    ])
    assert torch.allclose(output, ans)


@pytest.mark.parametrize("b,K,d", [(3, 1, 6), (1, 3, 10), (8, 5, 64)])
@pytest.mark.parametrize("T,t", [(3, 0), (3, 1), (3, 3), (5, 4)])
def test_pack_unpack(b: int, K: int, T: int, t: int, d: int):
    """
    Pack and then unpack states from parallel environments
    (same number of speakers, same number of requested words)
    """
    g_shape = (b, K, d) if K != 1 else (b, d)
    g = torch.randn(g_shape)
    x = torch.randn((b, t, d)) if t > 0 else None
    packed = envtools.pack_states(g, x, T)
    g_out, x_out, lengths = envtools.unpack_states(packed)
    t_out = lengths[0].item()
    assert (torch.allclose(g, g_out)
            and torch.all(lengths == t_out)
            and ((t == 0 and t_out == 0)
                 or torch.allclose(x, x_out[:, :t_out])))


@pytest.mark.parametrize("b", [1, 2, 8])
@pytest.mark.parametrize("d", [4, 16])
@pytest.mark.parametrize("K,T", [(1, 2), (5, 3), (10, 1)])
def test_unpack_from_buffer(b: int, K: int, T: int, d: int):
    g_shape = (b, K, d) if K != 1 else (b, d)
    g = torch.randn(g_shape)
    x = torch.randn((b, T, d))
    buffer = []
    for t in range(T + 1):
        buffer.append(envtools.pack_states(
            voice_prints=g,
            word_embeddings=None if t == 0 else x[:, :t],
            num_words=T
        ))
    buffer = torch.cat(buffer, 0)
    inds = torch.randperm(len(buffer))
    g_out, x_out, lengths = envtools.unpack_states(buffer[inds])
    x_out_list = [x_i[:t] for x_i, t in zip(x_out, lengths)]
    x_ans_list = [x[ind % b, :ind // b, :] for ind in inds]
    g_repeats = (T + 1, 1, 1) if K != 1 else (T + 1, 1)
    assert (torch.allclose(g.repeat(g_repeats)[inds], g_out)
            and all(torch.allclose(output, ans)
                    for output, ans in zip(x_out_list, x_ans_list)))


def test_append():
    K = 2
    T = 3
    d = 3
    g = torch.zeros((1, K, d))
    state = envtools.pack_states(g, None, num_words=T)
    for t in range(T):
        x = torch.ones((1, d)) * (t + 1)
        envtools.append_word_vectors(state, x, K, t)
    ans = torch.tensor([
        [
            [K, T, T],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ]
    ])
    assert torch.allclose(ans, state)
