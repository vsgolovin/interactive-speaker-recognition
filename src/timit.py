from typing import Optional, Sequence, Tuple
from pathlib import Path
import librosa
import pandas as pd
import soundfile as sf
from common import PathLike, KALDI_ROOT


class TimitCorpus:
    def __init__(self, path: PathLike):
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
        with open(txtfile, "w") as fout:
            for word, word_id in self.words.items():
                fout.write(f"{word} {word_id}\n")

    def _get_speaker_directory(self, speaker_id: str,
                               info: Optional[pd.Series] = None) -> Path:
        if info is None:
            info = self.spkrinfo.loc[speaker_id]
        subset_dir = self.root / ("TRAIN" if info["Use"] == "TRN" else "TEST")
        dr_dir = subset_dir / f"DR{info['DR']}"
        spkr_dir = dr_dir / (info["Sex"] + speaker_id)
        assert spkr_dir.exists(), f"{spkr_dir} not found"
        return spkr_dir

    def split_common_sentences(self, save_to: PathLike,
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

    def kaldi_data_prep(self, words_dir: PathLike,
                        output_dir: PathLike = "data/kaldi"):
        """
        Create files needed for extracting embeddings with Kaldi.
        """
        # extract single word recordings if not done previously
        words_dir = Path(words_dir)
        if not words_dir.exists():
            self.split_common_sentences(words_dir)

        # check for sph2pipe
        sph2pipe = KALDI_ROOT / "tools/sph2pipe_v2.5/sph2pipe"
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
        words_out_dir = output_dir / "words"
        words_out_dir.mkdir(exist_ok=True)
        wav_scp = open(words_out_dir / "wav.scp", "w")
        utt2spk = open(words_out_dir / "utt2spk", "w")
        spk2utt = open(words_out_dir / "spk2utt", "w")

        if not self.words:
            words = {}
            with open(words_dir / "WORDS.TXT", "r") as fin:
                for line in fin:
                    wid, word = line.rstrip().split(" ")
                    words[wid] = word
        else:
            words = self.words

        for spkr_id in self.spkrinfo.index:
            spkr_dir = words_dir / spkr_id
            spk2utt.write(spkr_id)
            for word_id in words.values():
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


def read_prompts(file: PathLike) -> dict:
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


def read_spkrsent(file: PathLike) -> dict:
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


def read_spkrinfo(file: PathLike) -> pd.DataFrame:
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


def read_wrd_file(file: PathLike) -> Sequence[Tuple[int, int, str]]:
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


if __name__ == "__main__":
    timit = TimitCorpus("data/TIMIT")
    timit.kaldi_data_prep("data/words", "data/kaldi")
