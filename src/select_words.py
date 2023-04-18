"""
Measure how guesser accuracy globally depends on the words used. More
concretely, for every word in vocabulary estimate probability that guesser
will choose the correct speaker given that word have been selected and other
words are chosen randomly.
"""


from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything
import click
from isr.data.timit import TimitXVectors, read_words_txt
from isr.nnet import Guesser


@click.command()
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--num-envs", "--batch-size", type=int, default=200,
              help="number of ISR environments (games) to run in parallel")
@click.option("--episodes", "--test-games", type=int, default=20000,
              help="total number of episodes (games) to run")
def main(seed: int, split_seed: int, num_speakers: int, num_words: int,
         num_envs: int, episodes: int):
    seed_everything(seed)
    dset = TimitXVectors(seed=split_seed)
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))

    # read file with words
    word_dict = read_words_txt("data/words/WORDS.TXT")
    # word_id -> word (string)
    word_ids = sorted(word_dict.keys())
    V = len(word_dict)

    # evaluate guesser on train and val subsets
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    for subset, ax in zip(["train", "val"], [ax1, ax2]):
        word_acc = evaluate(guesser, dset, subset, num_speakers, num_words,
                            num_envs, episodes)
        acc_dict = dict(zip(word_ids, word_acc))
        save_accuracies(acc_dict, f"output/word-acc_{subset}.csv")
        bar_plot(word_acc, ax, ylabel=f"{subset} accuracy")
    ax2.set_xticks(np.arange(V))
    ax2.set_xticklabels([word_dict[wid] for wid in word_ids],
                        rotation="vertical")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig("output/word-acc.png", dpi=75)


def evaluate(guesser: Guesser, dset: TimitXVectors, subset: str,
             num_speakers: int, num_words: int, num_envs: int,
             episodes: int) -> np.ndarray:
    # arrays for storing results
    VOCAB_SIZE = len(dset.words)
    # how many times every word was used
    words_used_count = np.zeros(VOCAB_SIZE, dtype=np.int_)
    # how many times word was used in a successful game
    words_success_count = np.zeros_like(words_used_count)

    # test guesser on `episodes` games
    sampled_episodes = 0
    guesser.eval()
    with tqdm(total=episodes) as pbar:
        while sampled_episodes < episodes:
            # sampling + guesser forward pass
            bs = min(num_envs, episodes - sampled_episodes)
            g, target_ids, targets = dset.sample_games(
                batch_size=bs,
                subset=subset,
                num_speakers=num_speakers
            )
            word_inds = torch.multinomial(
                torch.ones(VOCAB_SIZE).repeat((bs, 1)),
                num_samples=num_words
            )
            x = torch.stack(
                [dset.word_vectors[spkr][inds]
                 for spkr, inds in zip(target_ids, word_inds)],
                dim=0
            )
            with torch.no_grad():
                probs = guesser.forward(g, x)
                predictions = probs.argmax(dim=1)
                correct = (predictions == targets).numpy()

            # counting words and successes
            word_inds = word_inds.numpy()
            inds, counts = np.unique(word_inds, return_counts=True)
            words_used_count[inds] += counts
            inds, counts = np.unique(word_inds[correct], return_counts=True)
            words_success_count[inds] += counts

            sampled_episodes += bs
            pbar.update(bs)

    return words_success_count / words_used_count  # ~accuracy


def save_accuracies(data: dict, file: str):
    with open(file, "w") as fout:
        for k, v in data.items():
            fout.write(f"{k},{v}\n")


def bar_plot(heights, ax, ylabel="", **kwargs):
    assert np.all(heights <= 1.0)
    ax.bar(np.arange(len(heights)), heights, **kwargs)
    h_min = heights.min()
    y_min = max(h_min - (1 - h_min) * 0.2, 0.0)
    ax.set_ylim(y_min, 1.0)
    ax.set_ylabel(ylabel)
    return ax


if __name__ == "__main__":
    main()
