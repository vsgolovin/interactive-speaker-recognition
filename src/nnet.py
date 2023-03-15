from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Guesser(nn.Module):
    def __init__(self, emb_dim: int = 512, dropout: float = 0.5,
                 output_format: str = "logit"):
        super().__init__()
        hidden_dim = emb_dim * 2
        self.attention = AdditiveAttention(emb_dim, hidden_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_dim, 1)
        )
        assert output_format in ["logit", "logprob", "proba"]
        self.output_format = output_format
        self.f_out = {
            "logit": nn.Identity(),
            "logprob": nn.LogSoftmax(dim=-2),
            "proba": nn.Softmax(dim=-2)
        }

    def forward(self, X: Tensor, G: Tensor) -> Tensor:
        """
        Parameters
        ----------
        X : Tensor
            word X-Vectors, shape (batch, T, d)
        G : Tensor
            speaker voice prints, shape (batch, K, d)

        Notation same as in the paper: `T` is the number of word utterances
        used for classification, `K` is the number of speakers in the current
        pool, `d` is the embedding dimension.
        """
        T = X.size(1)
        K = G.size(1)

        # pool X-Vectors with attention conditioned on g_hat
        g_hat = torch.mean(G, dim=1, keepdim=True)  # average over speakers
        x_hat = self.attention(q=g_hat.repeat(1, T, 1), k=X, v=X)

        # concatenate x_hat with guest voice print
        logits = self.mlp(torch.cat([x_hat.repeat((1, K, 1)), G], dim=2))
        return self.f_out[self.output_format](logits).squeeze(2)


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
        v_hat = torch.sum(alpha * v, dim=-2, keepdim=True)
        return v_hat


class Enquirer(nn.Module):
    def __init__(self, emb_dim: int = 512, vocab_size: int = 20):
        super().__init__()
        self.start_token = nn.Parameter(
            torch.randn(emb_dim) / (emb_dim)**0.5,
            # don't remember where this initialization comes from
            requires_grad=True
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            bidirectional=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g_hat: Tensor, x: Optional[Tensor] = None) -> Tensor:
        """
        Input tensor shapes:
            g_hat  (batch, d)
            x      (seq, batch, d)
        """
        bs = g_hat.size(0)
        start = self.start_token.repeat((1, bs, 1))
        if x is None:
            inp = start
        else:
            inp = torch.cat([start, x], dim=0)
        last_output = self.lstm(inp)[0][-1]  # [batch, d * 2]
        logits = self.mlp(torch.cat([last_output, g_hat], 1))
        return self.softmax(logits)


if __name__ == "__main__":
    d = 8
    s = 5
    bs = 2
    enq = Enquirer(emb_dim=d, vocab_size=s)
    g_hat = torch.randn(bs, d)
    vocab = torch.randn((s, d))
    print("Vocabulary:", vocab)

    x = None
    for i in range(3):
        print(f"\nSTEP {i}")
        print("X-vectors:", x)
        probs = enq(g_hat, x)
        print("Enquirer output:", probs)

        inds = torch.argmax(probs, 1)
        print("Selecting words:", inds)
        new_words = (vocab[inds]).unsqueeze(0)
        if x is None:
            x = new_words
        else:
            x = torch.cat([x, new_words])
