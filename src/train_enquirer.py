from pathlib import Path
from typing import Tuple, Iterable
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
UPDATES_PER_EVAL = 50
TOTAL_UPDATES = 400
NUM_EPISODES = EPISODES_PER_UPDATE * TOTAL_UPDATES
BATCH_SIZE = 500
EPOCHS_PER_UPDATE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    output_dir = Path("output")
    dset = timit.TimitXVectors(seed=SPLIT_SEED)
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))
    env = IsrEnvironment(dset, guesser)
    ppo = PPO(512, len(dset.words), device=DEVICE)
    buffer = Buffer(num_words=NUM_WORDS)

    # evaluate SR system before training enquirer
    r_train, r_val = [evaluate(ppo, env, subset=subset)
                      for subset in ("train", "val")]
    print("Average reward (== accuracy) before training enquirer:")
    print(f"  train: {r_train:.3f}")
    print(f"  validation: {r_val:.3f}")

    # train enquirer
    avg_rewards = [r_val]
    ppo.train()
    with tqdm(desc="PPO training", total=NUM_EPISODES) as pbar:
        for i in range(TOTAL_UPDATES):
            # actual training
            for _ in range(BATCHES_PER_UPDATE):
                states = env.reset("train", batch_size=NUM_ENVS,
                                   num_speakers=NUM_SPEAKERS,
                                   num_words=NUM_WORDS)
                for _ in range(NUM_WORDS):
                    actions, probs, values = ppo.step(states)
                    new_states, rewards = env.step(actions)
                    buffer.append(states, actions, probs, rewards, values)
                    states = new_states
                pbar.update(NUM_ENVS)
            ppo.update(buffer, BATCH_SIZE, EPOCHS_PER_UPDATE)
            buffer.empty()

            # evaluate on validation set
            if (i + 1) % UPDATES_PER_EVAL == 0:
                avg_rewards.append(evaluate(ppo, env, "val",
                                            progress_bar=False))
                ppo.train()

    # evaluate SR system after training enquirer
    r_train = evaluate(ppo, env, subset="train")
    print("Average reward (== accuracy) after training enquirer:")
    print(f"  train: {r_train:.3f}")
    print(f"  validation: {avg_rewards[-1]:.3f}")

    # save actor and critic weights
    ppo.save(output_dir)

    # plot avg. reward on validation set
    assert len(avg_rewards) == TOTAL_UPDATES // UPDATES_PER_EVAL + 1
    eval_step = EPISODES_PER_UPDATE * UPDATES_PER_EVAL
    episode_count = np.arange(0, NUM_EPISODES + 1, eval_step)
    plt.figure()
    plt.plot(episode_count, avg_rewards, "bo-")
    plt.ylabel("Avg. reward on validation set")
    plt.xlabel("Episodes")
    plt.savefig(output_dir / "enquirer_training.png", dpi=75)


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
             parallel_envs: int = 50, progress_bar: bool = True) -> float:
    ppo.eval()
    all_rewards = np.zeros(episodes)

    if progress_bar:
        pbar = tqdm(desc=f"SR evaluation on {subset}", total=episodes)

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
            rewards.cpu().numpy()
        performed += cur_envs

        if progress_bar:
            pbar.update(cur_envs)

    if progress_bar:
        pbar.close()

    return np.mean(all_rewards)


if __name__ == "__main__":
    main()
