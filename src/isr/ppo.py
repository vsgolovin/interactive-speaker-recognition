from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.nn import functional as F
from isr.common import PathLike
from isr.envtools import unpack_states
from isr.nnet import Enquirer


class Buffer:
    def __init__(self, num_words: int, lambda_gae: float = 0.95,
                 gamma: float = 0.9):
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

    def forward(self, states: Tensor) -> Tensor:
        g, x = unpack_states(states)
        g_hat = torch.mean(g, dim=1)
        return self.model(g_hat, x)

    @torch.no_grad()
    def act(self, states: Tensor, past_actions: Optional[Tensor] = None
            ) -> Tuple[Tensor, Tensor]:
        "Return actions and their probabilities according to current policy"
        probs_full = self.forward(states)
        if self.training:
            actions = torch.multinomial(probs_full, num_samples=1).squeeze(1)
        else:
            if past_actions is not None:
                # make it impossible to select previously used action
                probs_full[
                    torch.arange(states.size(0)).view(-1, 1),
                    past_actions
                ] = 0
            actions = torch.argmax(probs_full, dim=1)
        probs = probs_full.gather(1, actions.view(-1, 1))
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
        entropy = -torch.sum(probs_full * torch.log(probs_full), 1)
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
                 device: Union[torch.device, str], lr_actor: float = 1e-4,
                 lr_critic: float = 1e-4, ppo_clip: float = 0.2,
                 grad_clip: Optional[float] = 1.0,
                 entropy: Optional[float] = 0.01):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.actor = Actor(input_size, num_actions).to(self.device)
        self.critic = Critic(input_size).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_critic)

        # hyperparameters
        self.grad_clip = grad_clip
        self.ppo_clip = ppo_clip
        self.entropy = entropy

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
        losses = {
            "surrogate": 0.0,
            "entropy": 0.0,
            "critic": 0.0
        }

        for _ in range(epochs):
            # iterate over whole buffer
            for batch in buffer.get(bs):
                s, a, p0, v_target, adv = [t.to(self.device) for t in batch]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # update actor
                p, entropy = self.actor.get_probs_entropy(s, a)
                frac = p.ravel() / p0
                loss_surrogate = -torch.min(
                    frac * adv,
                    torch.clamp(frac, 1 - self.ppo_clip, 1 + self.ppo_clip)
                    * adv
                ).mean()
                if self.entropy:
                    loss_entropy = -self.entropy * entropy.mean()
                else:
                    loss_entropy = 0.0
                loss_actor = loss_surrogate + loss_entropy
                self.actor_optim.zero_grad()
                loss_actor.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.grad_clip)
                self.actor_optim.step()

                # update critic
                v_pred = self.critic(s).squeeze(1)
                loss_critic = F.mse_loss(v_pred, v_target)
                self.critic_optim.zero_grad()
                loss_critic.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.grad_clip)
                self.critic_optim.step()

                # update losses for logging
                w = s.size(0) / (len(buffer) * epochs)
                losses["surrogate"] += loss_surrogate.item() * w
                if self.entropy:
                    losses["entropy"] += loss_entropy.item() * w
                losses["critic"] += loss_critic.item() * w

        losses["actor"] = losses["surrogate"] + losses["entropy"]
        return losses

    @torch.no_grad()
    def step(self, states: Tensor, past_actions: Optional[Tensor] = None
             ) -> Tuple[Tensor, Tensor, Tensor]:
        states = states.to(self.device)
        actions, probs = self.actor.act(states, past_actions)
        values = self.critic(states).squeeze(-1)
        return actions, probs, values

    def save(self, output_dir: PathLike):
        path = Path(output_dir)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")
