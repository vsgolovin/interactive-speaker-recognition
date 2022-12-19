from typing import Union
from pathlib import Path
import pandas as pd


PathLike = Union[Path, str]


class TimitCorpus:
    def __init__(self, path: PathLike):
        path = Path(path)
        assert path.exists() and path.is_dir(), f"Invalid path {path}"
        self.root = path
        pass


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
