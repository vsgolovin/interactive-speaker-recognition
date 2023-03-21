from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor


def pack_states(voice_prints: Tensor, word_embeddings: Optional[Tensor] = None,
                num_words: int = 3) -> Tensor:
    """
    Pack a batch of voice prints (g) and a batch of word embeddings into a
    batch of states (stack([info, g, x])).
    """
    # inspect input
    batch_size, num_speakers, d_emb = voice_prints.shape
    assert d_emb >= 3
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
    return g, x


def append_word_vectors(packed: Tensor, x: Tensor, num_speakers: int,
                        word_index: int):
    "Works inplace"
    assert torch.all(packed[:, 0, 2] == word_index)
    packed[:, 0, 2] = word_index + 1
    j = 1 + num_speakers + word_index
    packed[:, j, :] = x
