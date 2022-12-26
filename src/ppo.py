from typing import Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.nn import functional as F


LAMBDA = 0.95
GAMMA = 0.99
BATCH_SIZE = 64
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
CLIP = 0.2
ENTROPY_COEF = None
DEVICE = torch.device("cuda:0")


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []

    def __len__(self):
        return len(self.probs)

    def append(self, state: np.ndarray, action: int, proba: float,
               reward: float, value: float):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(proba)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_lambda_returns_and_gae(self, lam: float, gamma: float
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        returns = np.array(self.rewards)
        for i in reversed(range(1, len(self))):
            returns[i - 1] += gamma * (self.values[i] * (1 - lam)
                                       + returns[i] * lam)
        gae = returns - np.asarray(self.values)
        return returns, gae


class Buffer:
    def __init__(self, lam: float = LAMBDA, gamma: float = GAMMA):
        self.states = []
        self.actions = []
        self.probs = []
        self.returns = []
        self.gae = []
        self.lam = lam
        self.gamma = gamma

    def __len__(self):
        return len(self.actions)

    def append(self, traj: Trajectory):
        self.states += traj.states
        self.actions += traj.actions
        self.probs += traj.probs
        returns, gae = traj.compute_lambda_returns_and_gae(self.lam,
                                                           self.gamma)
        self.returns += list(returns)
        self.gae += list(gae)

    def get(self, bs: int = BATCH_SIZE, bs_min: int = 8):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        probs = np.array(self.probs, dtype=np.float32)
        returns = np.array(self.returns, dtype=np.float32)
        gae = np.array(self.gae, dtype=np.float32)

        N = len(states)
        inds = np.random.permutation(N)
        i = 0
        while i < N - bs_min:
            ix = inds[i:i + bs]
            yield states[ix], actions[ix], probs[ix], returns[ix], gae[ix]
            i += bs


class Actor(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.model = get_ppo_fc_model(input_size, num_actions)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @torch.no_grad()
    def act(self, states: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        "Return actions and their probabilities according to current policy"
        logits = self.model(states)
        probs_full = torch.softmax(logits, 1)
        actions = torch.multinomial(probs_full, 1)
        probs = probs_full.gather(1, actions)
        return actions.cpu().numpy().ravel(), probs.cpu().numpy().ravel()

    def get_probs_entropy(self, states: Tensor, actions: Tensor
                          ) -> Tuple[Tensor, Tensor]:
        """
        Return probabilities of actions acc. to current policy
        and entropy of every action distribution
        """
        probs_full = torch.softmax(self.model(states), 1)
        entropy = torch.sum(probs_full * torch.log(probs_full), 1)
        probs = probs_full.gather(1, actions.view(-1, 1))
        return probs, entropy

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)


class Critic(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.model = get_ppo_fc_model(input_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)


class PPO:
    "Proximal Policy Optimization with clipping"
    def __init__(self, input_size: int, num_actions: int):
        self.actor = Actor(input_size, num_actions).to(DEVICE)
        self.critic = Critic(input_size).to(DEVICE)
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)

    def update(self, buffer: Buffer, bs: int = BATCH_SIZE, epochs: int = 3):
        actor_losses = []
        critic_losses = []

        for _ in range(epochs):
            # iterate over whole buffer
            for batch in buffer.get(bs):
                s, a, p0, v_target, adv = map(
                    lambda x: torch.tensor(x, device=DEVICE), batch)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # update actor
                p, entropy = self.actor.get_probs_entropy(s, a)
                frac = p.ravel() / p0
                actor_loss = -torch.min(
                    frac * adv,
                    torch.clamp(frac, 1 - CLIP, 1 + CLIP) * adv
                ).mean()
                if ENTROPY_COEF:
                    actor_loss -= ENTROPY_COEF * entropy.mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # update critic
                v_pred = self.critic(s).squeeze(1)
                critic_loss = F.mse_loss(v_pred, v_target)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return np.array(actor_losses), np.array(critic_losses)

    @torch.no_grad()
    def step(self, states: np.ndarray):
        states = torch.tensor(states, device=DEVICE)
        actions, probs = self.actor.act(states)
        values = self.critic(states).squeeze(-1).cpu().numpy()
        return actions, probs, values

    def save(self, path: Union[str, Path]):
        path = Path(path)
        self.actor.save(path / "actor.pth")
        self.critic.save(path / "critic.pth")


def get_ppo_fc_model(input_size: int, output_size: int):
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, output_size)
    )
