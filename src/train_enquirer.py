from typing import Tuple, Iterable
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
from nnet import Enquirer, Guesser
import timit
from envtools import pack_states, unpack_states, append_word_vectors
from ppo import Buffer, PPO
from train_guesser import SPLIT_SEED


NUM_WORDS = 3
NUM_SPEAKERS = 5
NUM_ENVS = 33
BATCHES_PER_UPDATE = 10
EPISODES_PER_UPDATE = NUM_ENVS * BATCHES_PER_UPDATE
NUM_EPISODES = EPISODES_PER_UPDATE * 600
BATCH_SIZE = 500
EPOCHS_PER_UPDATE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dset = timit.TimitXVectors(seed=SPLIT_SEED)
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))
    env = IsrEnvironment(dset, guesser)
    ppo = PPO(512, len(dset.words), device=DEVICE)
    buffer = Buffer(num_words=NUM_WORDS)

    for subset in ("train", "val", "test"):
        avg_reward = evaluate(ppo, env, subset=subset)
        print(f"Average reward before training [{subset}]: {avg_reward}")

    episodes = 0
    ppo.train()
    with tqdm(total=NUM_EPISODES) as pbar:
        while episodes < NUM_EPISODES:
            for _ in range(BATCHES_PER_UPDATE):
                states = env.reset("train", batch_size=NUM_ENVS,
                                   num_speakers=NUM_SPEAKERS,
                                   num_words=NUM_WORDS)
                for _ in range(NUM_WORDS):
                    actions, probs, values = ppo.step(states)
                    new_states, rewards = env.step(actions)
                    buffer.append(states, actions, probs, rewards, values)
                    states = new_states
                episodes += NUM_ENVS
                pbar.update(NUM_ENVS)
            ppo.update(buffer, BATCH_SIZE, EPOCHS_PER_UPDATE)

            buffer.empty()

    for subset in ("train", "val", "test"):
        avg_reward = evaluate(ppo, env, subset=subset)
        print(f"Average reward before training [{subset}]: {avg_reward}")

    # save actor and critic weights
    ppo.save("output")


class IsrEnvironment:
    def __init__(self, dataset: timit.TimitXVectors, guesser: Guesser):
        self.dset = dataset
        self.guesser = guesser.eval()

        # environment settings, updated by `reset()`
        self.subset = None
        self.batch_size = None
        self.num_speakers = None
        self.num_words = None

        # environment state, updated by `reset()` and `step()`
        self.word_index = None
        self.speaker_ids = None
        self.targets = None
        self.states = None

    def reset(self, subset: str = "train", batch_size: int = 32,
              num_speakers: int = 5, num_words: int = 3) -> Tensor:
        "Returns state tensor"
        voice_prints, speaker_ids, targets = self.dset.sample_games(
            batch_size, subset, num_speakers)
        self.subset = subset
        self.batch_size = batch_size
        self.num_speakers = num_speakers
        self.num_words = num_words
        self.word_index = 0
        self.speaker_ids = speaker_ids
        self.targets = targets
        self.states = pack_states(voice_prints, None, num_words)
        return self.states

    def step(self, word_inds: Iterable[int]) -> Tuple[Tensor, Tensor]:
        "Returns state and reward"
        x = self.dset.get_word_embeddings(self.speaker_ids, word_inds)
        append_word_vectors(self.states, x, self.num_speakers, self.word_index)
        self.word_index += 1

        # intermediate steps
        if self.word_index < self.num_words:
            return self.states, torch.zeros((self.batch_size,))

        # final step => evaluate guesser
        g, x = unpack_states(self.states)
        with torch.no_grad():
            output = self.guesser(g, x)
            predictions = torch.argmax(output, 1)
            rewards = (predictions == self.targets).to(torch.float32)
            return self.states, rewards


def evaluate(ppo: Enquirer, env: IsrEnvironment,
             subset: str = "val", episodes: int = 20000,
             parallel_envs: int = 50) -> float:
    ppo.eval()

    all_rewards = np.zeros(episodes)
    performed = 0
    while performed < episodes:
        cur_envs = min(episodes - performed, parallel_envs)
        states = env.reset(subset, batch_size=cur_envs,
                           num_speakers=NUM_SPEAKERS, num_words=NUM_WORDS)
        for _ in range(NUM_WORDS):
            actions, _, _ = ppo.step(states)
            new_states, rewards = env.step(actions)
            states = new_states
        all_rewards[performed:performed + cur_envs] = \
            rewards.cpu().numpy()[:cur_envs]
        performed += NUM_ENVS

    return np.mean(all_rewards)


if __name__ == "__main__":
    main()
