from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
from pytorch_lightning import seed_everything
import click
from isr.data.timit import TimitXVectors
from isr.nnet import Guesser


NUM_SPEAKERS = 5
NUM_WORDS = 3


@click.command()
@click.option("-A/--all", "all_subsets", is_flag=True, default=False)
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("--num-envs", "--batch-size", type=int, default=200,
              help="number of ISR environments (games) to run in parallel")
@click.option("--episodes", "--test-games", type=int, default=20000,
              help="total number of episodes (games) to run")
@click.option("--random-agent", is_flag=True, default=False,
              help="sample words with a random agent")
@click.option("-k", "--agent-num-words", type=int, default=10,
              help="number of top scoring words agent will use")
@click.option("--nonuniform", is_flag=True, default=False,
              help="use higher-scoring words more often")
@click.option("--temperature", type=float, default=0.05,
              help="softmax temperature for converting scores into " +
              "probabilities")
def main(all_subsets: bool, seed: int, split_seed: int, num_envs: int,
         episodes: int, random_agent: bool, agent_num_words: int,
         nonuniform: bool, temperature: float):
    # load dataset and guesser
    seed_everything(seed)
    dset = TimitXVectors(seed=split_seed)
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))

    # read word scores
    if random_agent:
        agent = RandomAgent(20)
    else:
        word_scores = read_word_scores("models/word-acc_val.csv")
        agent = HeuristicAgent(
            word_scores,
            k=agent_num_words,
            nonuniform=nonuniform,
            temperature=temperature
        )

    # evaluate on different subsets
    subsets = ["test"]
    if all_subsets:
        subsets = ["train", "val"] + subsets
    for subset in subsets:
        acc = evaluate(guesser, dset, agent, subset, num_envs, episodes)
        print(f"Accuracy on {subset}: {acc}")


def read_word_scores(path: str):
    scores = []
    with open(path, "r") as fin:
        for line in fin:
            acc = line.rstrip().split(",")[1]
            scores.append(float(acc))
    return np.array(scores, dtype=float)


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


def evaluate(guesser: Guesser, dset, agent, subset, num_envs, episodes):
    sampled_episodes = 0
    accuracy = 0.0
    guesser.eval()
    with tqdm(total=episodes) as pbar:
        while sampled_episodes < episodes:
            bs = min(num_envs, episodes - sampled_episodes)
            g, target_ids, targets = dset.sample_games(
                batch_size=bs,
                subset=subset,
                num_speakers=NUM_SPEAKERS
            )
            word_inds = agent.sample(num_envs, NUM_WORDS)
            x = torch.stack(
                [dset.word_vectors[spkr][inds]
                 for spkr, inds in zip(target_ids, word_inds)],
                dim=0
            )
            with torch.no_grad():
                probs = guesser.forward(g, x)
                predictions = probs.argmax(dim=1)
                acc = (predictions == targets).float().mean().item()
            accuracy += acc * bs / episodes

            sampled_episodes += bs
            pbar.update(bs)

    return accuracy


if __name__ == "__main__":
    main()
