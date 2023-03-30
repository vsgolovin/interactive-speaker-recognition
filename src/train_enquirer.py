from pathlib import Path
from typing import Tuple, Iterable
from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor
from common import PathLike
from nnet import Enquirer, Guesser
import timit
from envtools import pack_states, unpack_states, append_word_vectors
from ppo import Buffer, PPO


NUM_WORDS = 3
NUM_SPEAKERS = 5
NUM_ENVS = 33
BATCHES_PER_UPDATE = 10
EPISODES_PER_UPDATE = NUM_ENVS * BATCHES_PER_UPDATE
NUM_EPISODES = EPISODES_PER_UPDATE * 300
BATCH_SIZE = 100
EPOCHS_PER_UPDATE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    dset = XVectorDataset()
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))
    env = IsrEnvironment(dset, guesser)
    ppo = PPO(512, len(dset.words), device=DEVICE)
    buffer = Buffer(num_words=NUM_WORDS)

    avg_reward = evaluate(ppo, env, subset="test")
    print(f"Average reward before training: {avg_reward}")

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

    avg_reward = evaluate(ppo, env, subset="test")
    print(f"Average reward after training: {avg_reward}")

    # save actor and critic weights
    ppo.save("output")


class XVectorDataset:
    def __init__(self, data_dir: PathLike = "./data", val_size: float = 0.2):
        data_dir = Path(data_dir)

        # read TIMIT info
        doc_dir = data_dir / "TIMIT/DOC"
        # prompt_id -> prompt
        self.prompts = timit.read_prompts(doc_dir / "PROMPTS.TXT")
        # speaker_id -> (prompt_ids)
        self.spkrsent = timit.read_spkrsent(doc_dir / "SPKRSENT.TXT")
        # dataframe with speaker info
        self.spkrinfo = timit.read_spkrinfo(doc_dir / "SPKRINFO.TXT")
        # common words: word_id -> word; sorted ids
        self.words = timit.read_words_txt(data_dir / "words/WORDS.TXT")
        self.word_ids = tuple(sorted(self.words.keys()))

        # split speakers into subsets
        self.speakers = {"train": [], "val": [], "test": []}
        for spkr_id, spkr_info in self.spkrinfo.iterrows():
            if spkr_info["Use"] == "TRN":
                if np.random.random() > val_size:
                    self.speakers["train"].append(spkr_id)
                else:
                    self.speakers["val"].append(spkr_id)
            else:
                assert spkr_info["Use"] == "TST", "SPKRINFO.TXT read error"
                self.speakers["test"].append(spkr_id)
        # cast to array for better indexing
        for subset in ("train", "val", "test"):
            self.speakers[subset] = np.array(self.speakers[subset])

        # load embeddings
        # speaker_id ("ABC0") -> speaker embedding
        xv_train = np.load(data_dir / "xvectors_train/spk_xvector.npz")
        xv_test = np.load(data_dir / "xvectors_test/spk_xvector.npz")
        # f"{speaker_id}_{word_id}" -> word embedding
        xv_words = np.load(data_dir / "xvectors_words/xvector.npz")
        # save embeddings dimension
        for vec in xv_train.values():
            self.emb_dim = vec.shape[0]
            break
        # copy embeddings to tensors, one per subset
        self.voice_prints = {}
        self.word_vectors = {}
        for subset in ("train", "val", "test"):
            spkrs = self.speakers[subset]
            xv = xv_test if subset == "test" else xv_train
            self.voice_prints[subset] = torch.zeros(
                size=(len(spkrs), self.emb_dim),
                dtype=torch.float32)
            for i, spkr in enumerate(spkrs):
                self.voice_prints[subset][i] = torch.FloatTensor(xv[spkr])
                keys = [f"{spkr}_{wid}" for wid in self.word_ids]
                self.word_vectors[spkr] = torch.FloatTensor(
                    np.stack([xv_words.get(key, np.zeros(self.emb_dim))
                             for key in keys]))

    def sample_games(self, batch_size: int, subset: str = "train",
                     num_speakers: int = 5
                     ) -> Tuple[Tensor, np.ndarray, Tensor]:
        # sample speakers for every game in batch
        spkr_inds = torch.multinomial(
            torch.ones(len(self.speakers[subset])).repeat((batch_size, 1)),
            num_samples=num_speakers)
        voice_prints = self.voice_prints[subset][spkr_inds, :]

        # select target speakers
        targets = torch.multinomial(
            torch.ones(num_speakers),
            num_samples=batch_size,
            replacement=True
        )
        # indices inside subset (integers)
        target_inds = spkr_inds[torch.arange(batch_size), targets]
        # speaker ids (strings)
        target_ids = self.speakers[subset][target_inds]

        return voice_prints, target_ids, targets

    def get_word_embeddings(self, speaker_ids: np.ndarray,
                            word_inds: Iterable[int]) -> Tensor:
        return torch.stack(
            [self.word_vectors[spkr][wrd.item()]
             for spkr, wrd in zip(speaker_ids, word_inds)],
            dim=0)


class IsrEnvironment:
    def __init__(self, dataset: XVectorDataset, guesser: Guesser):
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
