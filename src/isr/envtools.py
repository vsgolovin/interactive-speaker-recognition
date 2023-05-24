from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from isr import timit


def pack_states(voice_prints: Tensor, word_embeddings: Optional[Tensor] = None,
                num_words: int = 3) -> Tensor:
    """
    Pack a batch of voice prints (g) and a batch of word embeddings into a
    batch of states (stack([info, g, x])).
    """
    # inspect input
    if voice_prints.ndim == 2:  # 1 speaker => verification
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


def unpack_states(packed: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    "Extract voice prints and word embeddings from a batch of states"
    info = packed[:, 0, :3]
    num_speakers = int(info[0, 0].item())
    assert torch.all(info[:, 0] == num_speakers)
    num_words = int(info[0, 1].item())
    assert torch.all(info[:, 1] == num_words)
    num_req_words = info[:, 2].cpu().long()
    _, g, x = torch.split(
        packed,
        [1, num_speakers, num_words],
        dim=1
    )
    if num_speakers == 1:
        g = g.squeeze(1)
    return g, x, num_req_words


def append_word_vectors(packed: Tensor, x: Tensor, num_speakers: int,
                        word_index: int):
    "Works inplace"
    assert torch.all(packed[:, 0, 2] == word_index)
    packed[:, 0, 2] = word_index + 1
    j = 1 + num_speakers + word_index
    packed[:, j, :] = x


def g_to_g_hat(g: Tensor) -> Tensor:
    "Mean or identity depending on task"
    if g.ndim == 3:
        return torch.mean(g, dim=1)
    return g


class IsrEnvironment:
    def __init__(self, dataset: timit.TimitXVectors, guesser: nn.Module,
                 word_ind_transform: Optional[Tensor] = None):
        """
        `word_ind_transform` is a tensor of integers that maps words from
        word selection model (typically `Enquirer`) to dataset vocabulary.
        Default is `None`, which assumes no transformation is needed -- model
        and dataset have the same vocabulary
        """
        self.dset = dataset
        self.model = guesser.eval()
        self.transform = word_ind_transform

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
        self.noise_inds = None

        # method to initialize games
        self.sampler = self.dset.sample_isr_games

    def _init_game(self) -> Tuple[Tensor, np.ndarray, Tensor]:
        return self.dset.sample_isr_games(self.batch_size, self.subset,
                                          self.num_speakers)

    def reset(self, subset: str = "train", batch_size: int = 32,
              num_speakers: int = 5, num_words: int = 3,
              noise_ind: Optional[int] = None) -> Tensor:
        "Returns state tensor"
        self.subset = subset
        self.batch_size = batch_size
        self.num_speakers = num_speakers
        self.num_words = num_words
        self.word_index = 0
        voice_prints, speaker_ids, targets = self._init_game()
        self.speaker_ids = speaker_ids
        self.targets = targets
        if noise_ind is None:
            self.noise_inds = torch.randint(0, len(self.dset.noise_types),
                                            size=(batch_size,))
        else:
            self.noise_inds = torch.full((batch_size,), fill_value=noise_ind,
                                         dtype=torch.int64)
        self.states = pack_states(voice_prints, None, num_words)
        return self.states.clone()

    def _step(self, word_inds: Tensor) -> Tensor:
        word_inds = word_inds.cpu()
        if self.transform is not None:
            word_inds = self.transform[word_inds.to(self.transform.device)]
        x = self.dset.get_word_embeddings(self.speaker_ids, word_inds,
                                          noise_inds=self.noise_inds)
        append_word_vectors(self.states, x, self.num_speakers, self.word_index)
        self.word_index += 1
        return x

    def step(self, word_inds: Tensor) -> Tuple[Tensor, Tensor]:
        "Returns state and reward"
        x = self._step(word_inds)

        # intermediate steps
        if self.word_index < self.num_words:
            return self.states, torch.zeros((self.batch_size,))

        # final step => evaluate guesser
        g, x, lengths = unpack_states(self.states)
        assert torch.all(lengths == self.num_words)
        with torch.no_grad():
            output = self.model(g, x)
            predictions = torch.argmax(output, 1)
            rewards = (predictions == self.targets).to(torch.float32)
            return self.states.clone(), rewards


class IsvEnvironment(IsrEnvironment):
    def __init__(self, dataset: timit.TimitXVectors, verifier: nn.Module,
                 word_ind_transform: Optional[Tensor] = None):
        super().__init__(dataset, verifier, word_ind_transform)

    def reset(self, subset: str = "train", batch_size: int = 32,
              num_words: int = 3) -> Tensor:
        "Returns state tensor"
        return super().reset(subset, batch_size, 1, num_words)

    def _init_game(self) -> Tuple[Tensor, np.ndarray, Tensor]:
        return self.dset.sample_isv_games(self.batch_size, self.subset)

    def step(self, word_inds: Tensor) -> Tuple[Tensor, Tensor]:
        "Returns state and reward"
        x = self._step(word_inds)

        # intermediate steps
        if self.word_index < self.num_words:
            return self.states.clone(), torch.zeros((self.batch_size,))

        # final step => evaluate verifier
        g, x = unpack_states(self.states)
        with torch.no_grad():
            output = self.model(g, x)
            predictions = torch.round(output).long()
            rewards = (predictions == self.targets).float()
            return self.states.clone(), rewards
