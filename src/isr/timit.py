from typing import Iterable, Optional, Sequence, Tuple, Union
from pathlib import Path
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
import torchaudio


class TimitCorpus:
    def __init__(self, path: Union[Path, str]):
        path = Path(path)
        assert path.exists() and path.is_dir(), f"Invalid path {path}"
        self.root = path
        # prompt_id -> prompt
        self.prompts = read_prompts(self.root / "DOC/PROMPTS.TXT")
        # speaker_id -> (prompt_ids)
        self.spkrsent = read_spkrsent(self.root / "DOC/SPKRSENT.TXT")
        # dataframe with speaker info
        self.spkrinfo = read_spkrinfo(self.root / "DOC/SPKRINFO.TXT")
        # dict of words and their IDs
        self.words = {}

    def _get_word_id(self, word: str) -> str:
        wid = self.words.get(word, None)
        if wid is None:
            wid = f"WRD{len(self.words):02d}"
            self.words[word] = wid
        return wid

    def _save_word_ids(self, word_dir: Path):
        txtfile = word_dir / "WORDS.TXT"
        write_words_txt(self.words, txtfile)

    def _get_speaker_directory(self, speaker_id: str,
                               info: Optional[pd.Series] = None) -> Path:
        if info is None:
            info = self.spkrinfo.loc[speaker_id]
        subset_dir = self.root / ("TRAIN" if info["Use"] == "TRN" else "TEST")
        dr_dir = subset_dir / f"DR{info['DR']}"
        spkr_dir = dr_dir / (info["Sex"] + speaker_id)
        assert spkr_dir.exists(), f"{spkr_dir} not found"
        return spkr_dir

    def split_common_sentences(self, save_to: Union[Path, str],
                               drop: Sequence[str] = ["an"]) -> None:
        # output directory
        save_to = Path(save_to)
        assert not save_to.exists()
        save_to.mkdir(parents=True)

        # get waveforms of words from SA1 and SA2 for every speaker
        for spkr_id, spkr_info in self.spkrinfo.iterrows():
            spkr_inp_dir = self._get_speaker_directory(spkr_id, spkr_info)
            spkr_out_dir = save_to / spkr_id
            spkr_out_dir.mkdir()
            for pid in ("SA1", "SA2"):
                wfm, sr = librosa.load(spkr_inp_dir / f"{pid}.WAV", sr=16000)
                timestamps = read_wrd_file(spkr_inp_dir / f"{pid}.WRD")
                for start, end, word in timestamps:
                    word = word.lower()
                    if word in drop:
                        continue
                    wid = self._get_word_id(word)
                    sf.write(
                        file=spkr_out_dir / f"{wid}.WAV",
                        data=wfm[start:end],
                        samplerate=sr
                    )
        self._save_word_ids(save_to)  # writes to `save_to`/WORDS.TXT

    def add_noise_to_words(self, words_dir: Union[Path, str],
                           noise_file: Tensor, save_to: Union[Path, dir],
                           snr: int = 3):
        """
        Add noise from `noise_file` to word recordings in `words_dir`, save
        resulting .wav files to `save_to` directory.
        """
        # load noise
        noise, noise_sr = torchaudio.load(noise_file)
        sz_noise = noise.size(1)

        for speaker_dir in words_dir.iterdir():
            # every directory in words_dir will be created in save_to
            if not speaker_dir.is_dir():
                continue
            out_dir = save_to / speaker_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for file in speaker_dir.iterdir():
                if not file.suffix.lower() == ".wav":
                    continue
                wfm, sr = torchaudio.load(file)
                assert sr == noise_sr

                # select noise segment
                sz = wfm.size(1)
                assert sz <= sz_noise, f"{file} is longer than noise"
                start = random.randrange(0, sz_noise - sz)
                end = start + sz

                # add noise segment to waveform
                noisy = torchaudio.functional.add_noise(
                    wfm, noise[:, start:end], snr=torch.tensor([snr]))
                torchaudio.save(out_dir / file.name, noisy, sample_rate=sr,
                                encoding="PCM_S", bits_per_sample=16)

    def kaldi_data_prep(self, words_dir: Union[Path, str],
                        kaldi_root: Union[Path, str],
                        noise_names: Iterable[str] = [],
                        output_dir: Union[Path, str] = "data/kaldi"):
        """
        Create files needed for extracting embeddings with Kaldi.
        """
        kaldi_root = Path(kaldi_root)
        assert kaldi_root.exists() and kaldi_root.is_dir()

        # extract single word recordings if not done previously
        words_dir = Path(words_dir)
        if not words_dir.exists():
            self.split_common_sentences(words_dir)

        # check for sph2pipe
        sph2pipe = kaldi_root / "tools/sph2pipe_v2.5/sph2pipe"
        assert sph2pipe.exists()
        sph2pipe_str = str(sph2pipe) + " -f wav"

        # create wav.scp, utt2spk and spk2utt files for every subset
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        label2subset = {"TRN": "train", "TST": "test"}
        for subset in label2subset.values():
            subset_dir = output_dir / subset
            subset_dir.mkdir(exist_ok=True)

            # utterance id -> wav file location
            wav_scp = open(subset_dir / "wav.scp", "w")
            # utterance id -> speaker id and inverse
            utt2spk = open(subset_dir / "utt2spk", "w")
            spk2utt = open(subset_dir / "spk2utt", "w")

            for spkr_id, spkr_info in self.spkrinfo.iterrows():
                # check if speaker is in the current subset
                if label2subset[spkr_info["Use"]] != subset:
                    continue
                spkr_dir = self._get_speaker_directory(spkr_id, spkr_info)
                spk2utt.write(spkr_id)
                for sentence_id in sorted(self.spkrsent[spkr_id]):
                    utt_id = spkr_id + "_" + sentence_id
                    wav_file = spkr_dir / (sentence_id + ".WAV")
                    assert wav_file.exists()
                    wav_scp.write(" ".join(
                        [utt_id, sph2pipe_str, str(wav_file.absolute()), "|\n"]
                    ))
                    utt2spk.write(f"{utt_id} {spkr_id}\n")
                    spk2utt.write(f" {utt_id}")
                spk2utt.write("\n")

            wav_scp.close()
            utt2spk.close()
            spk2utt.close()

        # create the same files for single word recordings
        if not self.words:
            words = read_words_txt(words_dir / "WORDS.TXT")
        else:
            words = self.words
        for suffix in [""] + list(noise_names):
            self._kaldi_process_words(words_dir, words, output_dir,
                                      suffix=suffix)

    def _kaldi_process_words(self, words_dir: Path, words: dict,
                             output_dir: Path, suffix: str = ""):
        dirname = "words" if suffix == "" else f"words_{suffix}"
        words_dir = words_dir.parent / dirname
        words_out_dir = output_dir / dirname
        words_out_dir.mkdir(exist_ok=True)
        wav_scp = open(words_out_dir / "wav.scp", "w")
        utt2spk = open(words_out_dir / "utt2spk", "w")
        spk2utt = open(words_out_dir / "spk2utt", "w")

        for spkr_id in self.spkrinfo.index:
            spkr_dir = words_dir / spkr_id
            spk2utt.write(spkr_id)
            for word_id in words.keys():
                utt_id = spkr_id + "_" + word_id
                wav_file = spkr_dir / f"{word_id}.WAV"
                assert wav_file.exists()
                wav_scp.write(" ".join(
                    [utt_id, str(wav_file.absolute()), "\n"]
                ))
                utt2spk.write(f"{utt_id} {spkr_id}\n")
                spk2utt.write(f" {utt_id}")
            spk2utt.write("\n")

        wav_scp.close()
        utt2spk.close()
        spk2utt.close()


def read_prompts(file: Union[Path, str]) -> dict:
    """
    Read prompts from PROMPTS.TXT file.
    Return dictionary prompt_id -> prompt.
    """
    with open(file, "r") as fin:
        # skip header
        for _ in range(6):
            next(fin)
        # read prompts
        prompts = {}
        for line in fin:
            line = line.rstrip()
            if len(line) == 0:
                continue
            prompt, pid = line.split(" (")
            pid = pid[:-1]  # "sa1)" -> "sa1"
            prompts[pid] = prompt
    assert len(prompts) == 2342, f"Error: found {len(prompts)} prompts"
    return prompts


def read_spkrsent(file: Union[Path, str]) -> dict:
    """
    Read SPKRSENT.TXT to obtain sentence (prompt) ids for every speaker.
    Return dictionary speaker_id -> tuple of prompt_id's
    """
    with open(file, "r") as fin:
        # skip header
        for _ in range(10):
            next(fin)
        speaker_to_prompt = {}
        # line: speaker [SA] [SX] [SI]
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            line_lst = line.split()
            speaker_id = line_lst[0]
            # no need to save common SA1 and SA2
            prompt_ids = ["SX" + num for num in line_lst[3:8]]
            prompt_ids += ["SI" + num for num in line_lst[8:11]]
            speaker_to_prompt[speaker_id] = tuple(prompt_ids)
    return speaker_to_prompt


def read_spkrinfo(file: Union[Path, str]) -> pd.DataFrame:
    """
    Read SPKRINFO.TXT to obtain data about every speaker.
    """
    with open(file, "r") as fin:
        # skip header
        for _ in range(39):
            next(fin)
        # read the table
        data = []
        columns = ("ID", "Sex", "DR", "Use", "RecDate", "BirthDate", "Ht",
                   "Race", "Edu", "Comments")
        comment_index = len(columns) - 1
        for line in fin:
            line = line.rstrip()
            if len(line) == 0:
                continue
            lst = line.split()
            if len(lst) == comment_index:
                lst.append("")
            else:
                lst[comment_index] = " ".join(lst[comment_index:])
                lst = lst[:comment_index + 1]
            data.append(lst)

    # construct dataframe
    assert len(data) == 630, "Wrong speaker count"
    df = pd.DataFrame(data=data, columns=columns)
    return df.set_index("ID")


def read_wrd_file(file: Union[Path, str]) -> Sequence[Tuple[int, int, str]]:
    """
    Read a .WRD file with timestamps for every word in a recording.
    """
    timestamps = []
    with open(file, "r") as fin:
        for line in fin:
            lst = line.rstrip().split()
            if lst:
                start, end, word = lst
                timestamps.append((int(start), int(end), word))
    return timestamps


def write_words_txt(words: dict, txtfile: Path):
    with open(txtfile, "w") as fout:
        for word, word_id in words.items():
            fout.write(f"{word} {word_id}\n")


def read_words_txt(txtfile: Path) -> dict:
    words = {}
    with open(txtfile, "r") as fin:
        for line in fin:
            word, wid = line.rstrip().split(" ")
            words[wid] = word
    return words


class TimitXVectors:
    """
    Dataset of X-Vector embeddings for speakers, sentences and words.
    """
    def __init__(self, data_dir: Union[Path, str] = "./data",
                 val_size: float = 0.2, seed: Optional[int] = None,
                 noisy_words: bool = False):
        """

        Parameters
        ----------
        data_dir : Path | str
            Path to TIMIT dataset and extracted embeddings.
        val_size : float
            Fraction of train dataset (speakers) to put in the validation set.
        seed : int, Optional
            Seed for train / validation split. Uses `random` library and
            restores previous random state after performing split.

        """
        data_dir = Path(data_dir)

        # read TIMIT info
        doc_dir = data_dir / "TIMIT/DOC"
        # prompt_id -> prompt
        self.prompts = read_prompts(doc_dir / "PROMPTS.TXT")
        # speaker_id -> (prompt_ids)
        self.spkrsent = read_spkrsent(doc_dir / "SPKRSENT.TXT")
        # dataframe with speaker info
        self.spkrinfo = read_spkrinfo(doc_dir / "SPKRINFO.TXT")
        # common words: word_id -> word; sorted ids
        self.words = read_words_txt(data_dir / "words/WORDS.TXT")
        self.word_ids = tuple(sorted(self.words.keys()))
        self.vocab_size = len(self.words)
        self.noise_types = ["none"]
        if noisy_words:
            self.noise_types += sorted(
                [d.name.split("_")[1] for d in data_dir.glob("words_*")])

        # split speakers into subsets
        if seed is not None:
            cur_state = random.getstate()
            random.seed(seed)
        self.speakers = {"train": [], "val": [], "test": []}
        for spkr_id, spkr_info in self.spkrinfo.iterrows():
            if spkr_info["Use"] == "TRN":
                if random.random() > val_size:
                    self.speakers["train"].append(spkr_id)
                else:
                    self.speakers["val"].append(spkr_id)
            else:
                assert spkr_info["Use"] == "TST", "SPKRINFO.TXT read error"
                self.speakers["test"].append(spkr_id)

        # restore previous random state
        if seed:
            random.setstate(cur_state)

        # cast to array for better indexing
        for subset in ("train", "val", "test"):
            self.speakers[subset] = np.array(self.speakers[subset])

        # load embeddings
        # speaker_id ("ABC0") -> speaker embedding
        xv_train = np.load(data_dir / "xvectors_train/spk_xvector.npz")
        xv_test = np.load(data_dir / "xvectors_test/spk_xvector.npz")
        # f"{speaker_id}_{word_id}" -> word embedding
        xv_words = [np.load(data_dir / "xvectors_words/xvector.npz")]
        for noise_type in self.noise_types[1:]:
            xv_words.append(
                np.load(data_dir / f"xvectors_words_{noise_type}/xvector.npz")
            )
        # save embeddings dimension
        for vec in xv_train.values():
            self.emb_dim = vec.shape[0]
            break
        # copy embeddings to tensors, one per subset
        self.voice_prints = {}
        self.word_vectors = [{} for _ in range(len(self.noise_types))]
        for subset in ("train", "val", "test"):
            spkrs = self.speakers[subset]
            xv = xv_test if subset == "test" else xv_train
            self.voice_prints[subset] = torch.zeros(
                size=(len(spkrs), self.emb_dim),
                dtype=torch.float32)
            for i, spkr in enumerate(spkrs):
                self.voice_prints[subset][i] = torch.FloatTensor(xv[spkr])
                keys = [f"{spkr}_{wid}" for wid in self.word_ids]
                for j in range(len(self.noise_types)):
                    self.word_vectors[j][spkr] = torch.FloatTensor(
                        np.stack([xv_words[j].get(key, np.zeros(self.emb_dim))
                                  for key in keys]))

    def sample_isr_games(self, batch_size: int, subset: str = "train",
                         num_speakers: int = 5
                         ) -> Tuple[Tensor, np.ndarray, Tensor]:
        """
        Efficiently sample a batch of ISR games. Each game has the same amount
        of speakers.

        Returns
        -------
        voice_prints : Tensor
            Stack of speaker voice prints for every game.
            shape (batch_size, num_speakers, emb_dim)
        target_ids : np.ndarray
            IDs (strings, i.e., "ABC0") of selected speakers.
            shape (batch_size,)
        targets : Tensor
            Relative index (integer from [1, n_speakers]) of target speaker for
            every game.
            shape (batch_size,)

        """
        # sample speakers for every game in batch
        spkr_inds = torch.multinomial(
            torch.ones((batch_size, len(self.speakers[subset]))),
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

    def sample_isv_games(self, batch_size: int, subset: str = "train"
                         ) -> Tuple[Tensor, np.ndarray, Tensor]:
        """
        Efficiently sample a batch of ISV games.

        Returns
        -------
        voice_prints : Tensor
            Stack of speaker voice prints for every game.
            shape (batch_size, emb_dim)
        real_ids : np.ndarray
            IDs (strings, i.e., "ABC0") of selected speakers.
            shape (batch_size,)
        targets : Tensor
            Tensor of ones and zeros. Ones indicate matches between voice
            prints and real speaker ids (authentic speakers), zeros indicate
            the opposite (impostors).
            shape (batch_size,)

        """
        # sample speakers
        # two speakers per sample -- impostor and authentic
        spkr_inds = torch.multinomial(
            torch.ones((batch_size, len(self.speakers[subset]))),
            num_samples=2
        )
        targets = torch.randint(0, 2, size=(batch_size,))
        # speakers to verify (authentic)
        ver_inds = spkr_inds[:, 1]
        voice_prints = self.voice_prints[subset][ver_inds, :]
        # actual speakers, use their word embeddings (impostor or authentic)
        real_inds = spkr_inds.gather(1, targets.unsqueeze(1)).squeeze(1)
        # their ids (strings, not integers)
        real_ids = self.speakers[subset][real_inds]
        return voice_prints, real_ids, targets

    def get_word_embeddings(self, speaker_ids: np.ndarray,
                            word_inds: Tensor,
                            noise_inds: Optional[Tensor] = None) -> Tensor:
        if noise_inds is None:
            noise_inds = torch.randint(low=0, high=len(self.noise_types),
                                       size=(word_inds.size(0),))

        # TODO: store word embeddings differently to avoid using listcomp
        return torch.stack(
            [self.word_vectors[n][spkr][words]
             for n, spkr, words in zip(noise_inds, speaker_ids, word_inds)],
            dim=0)

    def create_codebook(self, subset: str = "train") -> Tensor:
        wv_stack = torch.stack([self.word_vectors[0][spkr]
                                for spkr in self.speakers[subset]])
        codebook = wv_stack.mean(0)
        return codebook
