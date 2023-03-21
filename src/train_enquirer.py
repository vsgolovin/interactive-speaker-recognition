from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from common import PathLike
import timit
from envtools import pack_states, append_word_vectors


NUM_WORDS = 3
NUM_SPEAKERS = 5


def main():
    b = 10

    dset = XVectorDataset()
    G, target_ids, targets = dset.sample_games(b, "train", NUM_SPEAKERS)
    print(G.shape)
    print(target_ids)
    print(targets)
    s = pack_states(G, None, NUM_WORDS)

    for i in range(NUM_WORDS):
        word_inds = torch.randint(0, len(dset.words), (b,))
        x = dset.get_word_embeddings(target_ids, word_inds)
        s = append_word_vectors(s, x, NUM_SPEAKERS, i)
        print(s[0, 0, :10])
        print(s.shape)


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

    def get_word_embeddings(self, speaker_ids: np.ndarray, word_inds: Tensor
                            ) -> Tensor:
        return torch.stack(
            [self.word_vectors[spkr][wrd]
             for spkr, wrd in zip(speaker_ids, word_inds)],
            dim=0)


if __name__ == "__main__":
    main()
