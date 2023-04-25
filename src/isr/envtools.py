from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from isr import timit
from isr.nnet import Guesser


def pack_states(voice_prints: Tensor, word_embeddings: Optional[Tensor] = None,
                num_words: int = 3) -> Tensor:
    """
    Pack a batch of voice prints (g) and a batch of word embeddings into a
    batch of states (stack([info, g, x])).
    """
    # inspect input
    if voice_prints.ndim == 2:
        batch_size, d_emb = voice_prints.shape
        num_speakers = 1
    else:
        batch_size, num_speakers, d_emb = voice_prints.shape
    if word_embeddings is None:
        num_requested_words = 0
    else:
        num_requested_words = word_embeddings.size(1)
    d_packed = 1 + num_speakers + num_words

    # pack data into a single tensor
    # every sample is a stack [info, voice_prints, word_embeddings]
    packed = torch.zeros((batch_size, d_packed, d_emb))
    packed[:, 0, 0] = num_speakers
    packed[:, 0, 1] = num_words
    packed[:, 0, 2] = num_requested_words
    j = num_speakers + 1
    if voice_prints.ndim == 2:
        packed[:, 1, :] = voice_prints
    else:
        packed[:, 1:j, :] = voice_prints
    if num_requested_words > 0:
        packed[:, j:j + num_requested_words, :] = word_embeddings

    return packed


def unpack_states(packed: Tensor) -> Tuple[Tensor, Tensor]:
    "Extract voice prints and word embeddings from a batch of states"
    num_speakers, num_words, num_req_words = packed[0, 0, :3] \
        .cpu().numpy().astype(np.int64)
    _, g, x, _ = torch.split(
        packed,
        [1, num_speakers, num_req_words, num_words - num_req_words],
        dim=1
    )
    if num_req_words == 0:
        x = None
    if num_speakers == 1:
        g = g.squeeze(1)
    return g, x


def append_word_vectors(packed: Tensor, x: Tensor, num_speakers: int,
                        word_index: int):
    "Works inplace"
    assert torch.all(packed[:, 0, 2] == word_index)
    packed[:, 0, 2] = word_index + 1
    j = 1 + num_speakers + word_index
    packed[:, j, :] = x


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
        voice_prints, speaker_ids, targets = self.dset.sample_isr_games(
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

    def step(self, word_inds: Tensor) -> Tuple[Tensor, Tensor]:
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
