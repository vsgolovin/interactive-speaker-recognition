from typing import Union
from pathlib import Path


PathLike = Union[Path, str]


class TimitCorpus:
    def __init__(self, path: PathLike):
        path = Path(path)
        assert path.exists() and path.is_dir(), f"Invalid path {path}"
        self.root = path
        pass


def load_prompts(file: PathLike) -> dict:
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
