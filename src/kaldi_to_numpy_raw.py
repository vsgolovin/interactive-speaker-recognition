"""
Script for importing raw x-vectors from `.ark` and `.scp` files.
"""

from typing import Iterable, Tuple
import numpy as np
from kaldi_io import read_vec_flt_scp
from common import PathLike


def read_vectors(scp_file: PathLike) -> Tuple[Tuple[str], Tuple[np.ndarray]]:
    "Read Kaldi vectors stored on disk"
    return zip(*read_vec_flt_scp(str(scp_file)))


def get_speaker_embeddings(keys: Iterable[str],
                           embeddings: Iterable[np.ndarray]) -> dict:
    speaker_ids = np.array([k.split("_")[0] for k in keys])
    speakers = np.unique(speaker_ids)
    embeddings = np.asarray(embeddings)
    out = {}
    for speaker in speakers:
        mask = (speaker_ids == speaker)
        out[speaker] = embeddings[mask].mean(0)
    return out


if __name__ == "__main__":
    from pathlib import Path

    # speaker embeddings
    for subset in ["train", "test"]:
        subset_dir = Path(f"data/xvectors_{subset}")
        keys, embeddings = read_vectors(subset_dir / "xvector.scp")
        data = get_speaker_embeddings(keys, embeddings)
        np.savez(subset_dir / "spk_xvector.npz", **data)

    # word embeddings
    subset_dir = Path("data/xvectors_words")
    keys, embeddings = read_vectors(subset_dir / "xvector.scp")
    data = dict(zip(keys, embeddings))
    np.savez(subset_dir / "xvector.npz", **data)
