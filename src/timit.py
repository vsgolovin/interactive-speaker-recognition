from typing import Optional, Sequence, Tuple, Union
from pathlib import Path
import librosa
import pandas as pd
import soundfile as sf


PathLike = Union[Path, str]


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

    def _get_speaker_directory(self, speaker_id: str,
                               info: Optional[pd.Series] = None) -> Path:
        if info is None:
            info = self.spkrinfo.loc[speaker_id]
        subset_dir = self.root / ("TRAIN" if info["Use"] == "TRN" else "TEST")
        dr_dir = subset_dir / f"DR{info['DR']}"
        spkr_dir = dr_dir / (info["Sex"] + speaker_id)
        assert spkr_dir.exists(), f"{spkr_dir} not found"
        return spkr_dir

    def split_common_sentences(self, save_to: PathLike) -> None:
        # output directory
        save_to = Path(save_to)
        assert save_to.exists() and save_to.is_dir()

        # get waveforms of words from SA1 and SA2 for every speaker
        for spkr_id, spkr_info in self.spkrinfo.iterrows():
            spkr_inp_dir = self._get_speaker_directory(spkr_id, spkr_info)
            spkr_out_dir = save_to / spkr_id
            spkr_out_dir.mkdir()
            for pid in ("SA1", "SA2"):
                wfm, sr = librosa.load(spkr_inp_dir / f"{pid}.WAV", sr=16000)
                timestamps = read_wrd_file(spkr_inp_dir / f"{pid}.WRD")
                for start, end, word in timestamps:
                    sf.write(
                        file=spkr_out_dir / f"{word.upper()}.WAV",
                        data=wfm[start:end],
                        samplerate=sr
                    )


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
    timit.split_common_sentences("data/words")
