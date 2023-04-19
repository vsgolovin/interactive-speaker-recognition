from pathlib import Path
from typing import Iterable, Tuple, Union
import click
from kaldi_io import read_vec_flt_scp
import numpy as np
from isr import timit


DATA_DIR = Path("./data")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("kaldi_root", nargs=1, type=click.Path())
def kaldi_data_prep(kaldi_root):
    data = timit.TimitCorpus(DATA_DIR / "TIMIT")
    data.kaldi_data_prep(
        words_dir=DATA_DIR / "words",
        kaldi_root=kaldi_root,
        output_dir=DATA_DIR / "kaldi"
    )


@cli.command()
def kaldi_to_numpy():
    "Convert kaldi embeddings to numpy arrays"
    # speaker embeddings
    for subset in ["train", "test"]:
        subset_dir = DATA_DIR / f"xvectors_{subset}"
        keys, embeddings = read_vectors(subset_dir / "xvector.scp")
        data = get_speaker_embeddings(keys, embeddings)
        np.savez(subset_dir / "spk_xvector.npz", **data)

    # word embeddings
    subset_dir = DATA_DIR / "xvectors_words"
    keys, embeddings = read_vectors(subset_dir / "xvector.scp")
    data = dict(zip(keys, embeddings))
    np.savez(subset_dir / "xvector.npz", **data)


def read_vectors(scp_file: Union[Path, str]
                 ) -> Tuple[Tuple[str], Tuple[np.ndarray]]:
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
    cli()
