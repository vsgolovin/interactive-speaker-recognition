from typing import Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.nn import functional as F
from common import PathLike
from envtools import unpack_states
from nnet import Enquirer


LAMBDA = 0.95
GAMMA = 0.9
ACTOR_LR = 5e-3
CRITIC_LR = 5e-3
GRAD_CLIP = 1.0
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01


class Buffer:
    def __init__(self, num_words: int, lambda_gae: float = LAMBDA,
                 gamma: float = GAMMA):
        self.T = num_words
        self.lam = lambda_gae
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.batch_size = None

    def __len__(self) -> int:
        return sum(s.size(0) for s in self.states)

    def append(self, states: Tensor, actions: Tensor, probs: Tensor,
               rewards: Tensor, values: Tensor):
        if self.batch_size is None:
            self.batch_size = states.size(0)
        for t in (states, actions, probs, rewards, values):
            assert t.size(0) == self.batch_size
        self.states.append(states.clone().cpu())
        self.actions.append(actions.clone().cpu())
        self.probs.append(probs.clone().cpu())
        self.rewards.append(rewards.clone().cpu())
        self.values.append(values.clone().cpu())

    def construct_tensors(self):
        # list -> tensor
        assert len(self.states) % self.T == 0
        s, a, p, r, v = [
            torch.cat(lst, dim=0)
            for lst in (self.states, self.actions, self.probs, self.rewards,
                        self.values)
        ]

        # compute lambda-returns and advantage
        start = self.batch_size * (self.T - 1)
        end = start + self.batch_size  # == self.T * self.batch_size
        n_games = s.size(0) // end
        inds = torch.cat(
            [torch.arange(start, start + self.batch_size) + end * i
             for i in range(n_games)],
            dim=0
        )  # indices of terminal states
        g = r  # transform reward tensor into lambda-returns
        for _ in range(self.T - 1):
            g[inds - self.batch_size] += \
                self.gamma * (v[inds] * (1 - self.lam) + g[inds] * self.lam)
            inds -= self.batch_size
        adv = g - v

        return s, a, p, g, adv

    def get(self, batch_size: int, batch_size_min: int = 8):
        s, a, p, g, adv = self.construct_tensors()
        N = s.size(0)
        inds = np.random.permutation(N)
        i = 0
        while i < N - batch_size_min:
            ix = inds[i:i + batch_size]
            yield s[ix], a[ix], p[ix], g[ix], adv[ix]
            i += batch_size

    def empty(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []


class Actor(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.model = Enquirer(emb_dim=input_size, n_outputs=num_actions)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @torch.no_grad()
    def act(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        "Return actions and their probabilities according to current policy"
        g, x = unpack_states(states)
        g_hat = torch.mean(g, dim=1)
        logits = self.model(g_hat, x)
        probs_full = torch.softmax(logits, 1)
        actions = torch.multinomial(probs_full, 1)
        probs = probs_full.gather(1, actions)
        return actions, probs

    def get_probs_entropy(self, states: Tensor, actions: Tensor
                          ) -> Tuple[Tensor, Tensor]:
        """
        Return probabilities of actions acc. to current policy
        and entropy of every action distribution
        """
        g, x = unpack_states(states)
        g_hat = torch.mean(g, dim=1)
        probs_full = torch.softmax(self.model(g_hat, x), 1)
        entropy = torch.sum(probs_full * torch.log(probs_full), 1)
        probs = probs_full.gather(1, actions.view(-1, 1))
        return probs, entropy


class Critic(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.model = Enquirer(emb_dim=input_size, n_outputs=1)

    def forward(self, states: Tensor) -> Tensor:
        g, x = unpack_states(states)
        g_hat = torch.mean(g, dim=1)
        return self.model(g_hat, x)


class PPO:
    "Proximal Policy Optimization with clipping"
    def __init__(self, input_size: int, num_actions: int,
                 device: Union[torch.device, str]):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.actor = Actor(input_size, num_actions).to(self.device)
        self.critic = Critic(input_size).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)

    def train(self):
        self.actor.train()
        self.critic.train()
        return self

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        return self

    def update(self, buffer: Buffer, bs: int, epochs: int
               ) -> Tuple[float, float]:
        actor_losses = []
        critic_losses = []

        for _ in range(epochs):
            # iterate over whole buffer
            for batch in buffer.get(bs):
                s, a, p0, v_target, adv = [t.to(self.device) for t in batch]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # update actor
                p, entropy = self.actor.get_probs_entropy(s, a)
                frac = p.ravel() / p0
                actor_loss = -torch.min(
                    frac * adv,
                    torch.clamp(frac, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv
                ).mean()
                if ENTROPY_COEF:
                    actor_loss -= ENTROPY_COEF * entropy.mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                if GRAD_CLIP:
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), GRAD_CLIP)
                self.actor_optim.step()

                # update critic
                v_pred = self.critic(s).squeeze(1)
                critic_loss = F.mse_loss(v_pred, v_target)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                if GRAD_CLIP:
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), GRAD_CLIP)
                self.critic_optim.step()

                actor_losses.append(actor_loss.item() * s.size(0))
                critic_losses.append(critic_loss.item() * s.size(0))

        n_samples = len(buffer)
        actor_loss = np.sum(actor_losses) / n_samples
        critic_loss = np.sum(critic_losses) / n_samples
        return actor_loss, critic_loss

    @torch.no_grad()
    def step(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        states = states.to(self.device)
        actions, probs = self.actor.act(states)
        values = self.critic(states).squeeze(-1)
        return actions, probs, values

    def save(self, output_dir: PathLike):
        path = Path(output_dir)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")
