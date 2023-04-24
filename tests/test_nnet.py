from typing import Union
import pytest
import numpy as np
import torch
from isr import nnet


@pytest.mark.parametrize("b,t,k,d", [(1, 3, 5, 16), (32, 5, 10, 128)])
def test_guesser(b: int, t: int, k: int, d: int):
    X = torch.randn((b, t, d))
    G = torch.randn((b, k, d))
    guesser = nnet.Guesser(emb_dim=d, output_format="proba")
    probs = guesser(G, X)
    prob_sums = probs.detach().sum(1).numpy()
    assert probs.size() == torch.Size((b, k)) \
        and np.allclose(prob_sums, np.ones(b))


@pytest.mark.parametrize("b,t,d", [(1, 3, 16), (16, 5, 128)])
def test_additive_attention(b: int, t: int, d: int):
    q = torch.zeros((b, t, d))
    k = torch.zeros((b, t, d))
    v = torch.zeros((b, t, d))

    att = nnet.AdditiveAttention(d, d * 2)
    v_hat = att(q, k, v)
    assert v_hat.size() == torch.Size((b, 1, d))


@pytest.mark.parametrize("L", (0, 1, 3))
@pytest.mark.parametrize("b,d,v", [(1, 16, 3), (16, 128, 20)])
def test_enquirer(b: int, d: int, v: int, L: Union[int, None]):
    enq = nnet.Enquirer(emb_dim=d, n_outputs=v)
    g_hat = torch.randn((b, d))
    x = torch.randn((b, L, d)) if L > 0 else None
    probs = enq(g_hat, x)
    assert probs.shape == torch.Size([b, v]) and \
        torch.allclose(probs.sum(1), torch.ones(b)) and \
        torch.all((probs > 0) & (probs < 1))


@pytest.mark.parametrize("backend", ["mlp", "cs"])
@pytest.mark.parametrize("b,t,d", [(1, 1, 16), (16, 5, 128)])
def test_verifier(b: int, t: int, d: int, backend: str):
    model = nnet.Verifier(emb_dim=d, backend=backend)
    g = torch.randn((b, d))
    x = torch.randn((b, t, d))
    probs = model(g, x)
    assert probs.shape == torch.Size([b]) and \
        torch.all(probs <= 1) and \
        torch.all(probs >= 0)
