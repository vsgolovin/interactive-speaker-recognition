from typing import Optional
import numpy as np
import torch


class HeuristicAgent:
    def __init__(self, word_scores: np.ndarray, k: Optional[int],
                 nonuniform: bool = False, temperature: float = 1.0):
        """
        Agent for selecting words irrespective of context by using global word
        scores.

        Parameters
        ----------
        word_scores : np.ndarray
            An array of scores for every word in a dictionary. Higher score
            corresponds to higher probability to select word.
        k : int | None
            Number of words with highest scores agent will actually select.
            `None` is equal to selecting `k` equal to total number of words,
            i.e., no words will be excluded.
        nonuniform : bool
            Whether to use scores during sampling. This parameter is `False`
            by default, which means that agent will sample uniformly among
            `k` words with the highest scores. If `True`, sampling will be
            performed with probabilities obtained by taking a softmax of
            scores.
        temperature : float
            Softmax temperature used to convert scores into probabilites.
            Ignored if `nonuniform=False`. Higher temperatures correspond to
            more uniform distributions.

        """
        assert word_scores.ndim == 1
        V = len(word_scores)
        if k is None or k > V:
            k = V
        self.k = k

        # indices of k top-scoring words
        self.words = torch.LongTensor(
            np.argsort(word_scores)[:-(k + 1):-1].copy())

        # sampling probabilities
        if nonuniform:
            s = torch.tensor(word_scores)[self.words] / temperature
            self.probs = torch.softmax(s, 0)
            entropy = -torch.sum(self.probs * torch.log(self.probs))
            print(f"Using nonuniform sampling with entropy {entropy:.3f}")
        else:
            self.probs = torch.ones(self.k)

    def sample(self, num_envs: int, num_words: int) -> torch.Tensor:
        assert num_words <= self.k
        inds = torch.multinomial(
            input=self.probs.repeat((num_envs, 1)),
            num_samples=num_words,
            replacement=False
        )
        return self.words[inds]


class RandomAgent:
    def __init__(self, total_words: int):
        "Uniform sampling from `total_words` words."
        self.V = total_words

    def sample(self, num_envs: int, num_words: int) -> torch.Tensor:
        assert num_words <= self.V
        return torch.multinomial(
            torch.ones(num_envs, self.V),
            num_samples=num_words,
            replacement=False
        )
