from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from isr import utils


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

    def forward(self, G: Tensor, X: Tensor) -> Tensor:
        """
        Parameters
        ----------
        G : Tensor
            speaker voice prints, shape (batch, K, d)
        X : Tensor
            word X-Vectors, shape (batch, T, d)

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
                 dropout: float = 0.1, keepdim: bool = True):
        super().__init__()
        self.keepdim = keepdim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        e = self.mlp(torch.cat([q, k], dim=-1))
        alpha = F.softmax(e, dim=-2)
        v_hat = torch.sum(alpha * v, dim=-2, keepdim=self.keepdim)
        return v_hat


class Enquirer(nn.Module):
    def __init__(self, emb_dim: int = 512, out_dim: int = 20):
        "Use n_outputs=1 for value function / critic"
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.start_token = nn.Parameter(
            torch.rand(emb_dim) / (emb_dim)**0.5,
            # same as LSTM weight initialization
            requires_grad=True
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            batch_first=True,
            bidirectional=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, out_dim),
            nn.Softmax(dim=-1) if out_dim > 1 else nn.Sigmoid()
        )

    def forward(self, G_hat: Tensor, X: Optional[Tensor] = None) -> Tensor:
        bs = G_hat.size(0)
        start = self.start_token.repeat((bs, 1, 1))
        if X is None:
            inp = start
        else:
            inp = torch.cat([start, X], dim=1)
        last_output = self.lstm(inp)[0][:, -1, :]  # [batch, d * 2]
        return self.mlp(torch.cat([last_output, G_hat], 1))


class CodebookEnquirer(Enquirer):
    def __init__(self, codebook_size: int, emb_dim: int):
        """
        Enquirer modification that uses codebook -- a tensor of word
        embeddings -- in order to compute final probabilities.
        By default codebook is randomly initialized, use `load_codebook()`
        in order to use word embeddings.

        Parameters
        ----------
        vocab_size : int
            Number of words in codebook.
        emb_dim : int
            Word embeddings dimension size.

        """
        super().__init__(emb_dim=emb_dim, out_dim=emb_dim)
        self.mlp.pop(-1)  # remove softmax
        self.cb_size = codebook_size
        self.emb_dim = emb_dim

        # random codebook
        codebook = torch.randn((self.cb_size, emb_dim))
        self.register_buffer("codebook", codebook, persistent=False)

        # codebook shift and scale for word embeddings normalization
        self.register_buffer("mu", torch.zeros((emb_dim,)), persistent=True)
        self.register_buffer("sigma", torch.ones((emb_dim,)), persistent=True)

        # softmax with trainable tempearature
        self.softmax = nn.Softmax(dim=1)
        self.t_coeff = nn.Parameter(torch.tensor(1e-4), requires_grad=True)

    @property
    def temperature(self):
        return torch.exp(self.t_coeff)

    def load_codebook(self, codebook: Tensor, update_stats: bool = False):
        """
        Parameters
        ----------
        codebook : Tensor
            2d tensor of word embeddings as rows.
        update_stats : bool
            Whether to use passed codebook mean and std for scaling. Default is
            `False`, which means scaling with currently stored values.

        """
        assert codebook.shape == torch.Size([self.cb_size, self.emb_dim])
        if update_stats:
            self.mu = codebook.mean(0)
            self.sigma = codebook.std(0)
        self.codebook = (codebook - self.mu) / self.sigma

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        word_emb = super().forward(g, x)
        distances = utils.pairwise_mse(word_emb, self.codebook)
        return self.softmax(-distances / self.temperature)


class Verifier(nn.Module):
    "Simple modification of `Guesser` for verification"
    def __init__(self, emb_dim: int = 512, dropout: float = 0.5,
                 backend: str = "mlp"):
        "Backend should be one of `('mlp', 'cs')`"
        super().__init__()
        hidden_dim = emb_dim * 2
        self.attention = AdditiveAttention(emb_dim, hidden_dim, dropout,
                                           keepdim=False)
        assert backend in ("mlp", "cs"), f"Unknown backend {backend}"
        self.backend_type = backend
        if backend == "mlp":
            self.backend = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(2 * hidden_dim, 1),
                nn.Flatten(0)
            )
        else:
            self.backend = CosineSimilarity(dim=1)

    def forward(self, g: Tensor, X: Tensor) -> Tensor:
        # pool x-vectors with attention conditioned on g
        T = X.size(1)
        x_hat = self.attention(q=g.unsqueeze(1).repeat(1, T, 1), k=X, v=X)

        # compare x_hat to speaker embedding
        if self.backend_type == "mlp":
            scores = self.backend(torch.cat([x_hat, g], dim=1))
        else:
            scores = self.backend(x_hat, g)
        return torch.sigmoid(scores)


class CosineSimilarity(nn.Module):
    "CosineSimilarity + Linear"
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cs = nn.CosineSimilarity(*args, **kwargs)
        self.fc = nn.Linear(1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        similarities = self.cs(x1, x2)
        return self.fc(similarities.unsqueeze(1)).squeeze(1)
