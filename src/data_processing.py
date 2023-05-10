from pathlib import Path
import os
from typing import Iterable, Tuple, Union
from tqdm import tqdm
import click
from kaldi_io import read_vec_flt_scp
import numpy as np
from isr import timit


DATA_DIR = Path("./data")
NOISE_DICT = {544: "rain", 489: "car", 334: "crowd", 435: "typing", 587: "hum",
              20: "white"}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--snr", type=int, default=3, help="signal-to-noise ratio")
def add_noise(snr: int):
    data = timit.TimitCorpus(DATA_DIR / "TIMIT")
    words_dir = DATA_DIR / "words"
    if not words_dir.exists():
        data.split_common_sentences(words_dir)
    for num, label in tqdm(NOISE_DICT.items()):
        data.add_noise_to_words(
            words_dir=words_dir,
            noise_file=DATA_DIR / "noise" / f"noise-free-sound-{num:04d}.wav",
            save_to=DATA_DIR / f"words_{label}",
            snr=snr
        )


@cli.command()
@click.option("--noise", is_flag=True, default=False,
              help="include noisy word recordings")
def kaldi_data_prep(noise: bool):
    kaldi_root = os.environ["KALDI_ROOT"]
    data = timit.TimitCorpus(DATA_DIR / "TIMIT")
    data.kaldi_data_prep(
        words_dir=DATA_DIR / "words",
        kaldi_root=kaldi_root,
        output_dir=DATA_DIR / "kaldi",
        noise_names=NOISE_DICT.values() if noise else []
    )


@cli.command()
@click.option("--noise", is_flag=True, default=False,
              help="include noisy word recordings")
def kaldi_to_numpy(noise: bool):
    "Convert kaldi embeddings to numpy arrays"
    # speaker embeddings
    for subset in ["train", "test"]:
        subset_dir = DATA_DIR / f"xvectors_{subset}"
        keys, embeddings = read_vectors(subset_dir / "xvector.scp")
        data = get_speaker_embeddings(keys, embeddings)
        np.savez(subset_dir / "spk_xvector.npz", **data)

    # word embeddings
    suffix_list = [""]
    if noise:
        suffix_list += sorted(NOISE_DICT.values())
    for suffix in suffix_list:
        dirname = "xvectors_words"
        if suffix:
            dirname += "_" + suffix
        subset_dir = DATA_DIR / dirname
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
