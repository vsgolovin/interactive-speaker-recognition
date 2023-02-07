import pytest
import torch
from src import nnet


@pytest.mark.parametrize("t,d", [(3, 16), (5, 128)])
def test_additive_attention(t: int, d: int):
    b, t, d = 4, 5, 16
    q = torch.zeros((b, t, d))
    k = torch.zeros((b, t, d))
    v = torch.zeros((b, t, d))

    att = nnet.AdditiveAttention(d, d * 2)
    v_hat = att(q, k, v)
    assert v_hat.size() == torch.Size((b, d))
