import pytest
import torch
from isr.cpc import CPC


@pytest.mark.parametrize("b,d,L", [(1, 512, 2048), (8, 32, 20480)])
@pytest.mark.parametrize("enc", ["ResnetEncoder", "LayerNormEncoder"])
@pytest.mark.parametrize("rnn", ["GRU", "LSTM"])
def test_cpc(b: int, d: int, L: int, enc: str, rnn: str):
    inp = torch.randn((b, 1, L))
    net = CPC(emb_dim=d, context_dim=d // 2, encoder=enc)
    z_emb, c_emb = net(inp)
    assert z_emb.size() == torch.Size([b, L // 160, d]) and \
        c_emb.size() == torch.Size([b, L // 160, d // 2])
