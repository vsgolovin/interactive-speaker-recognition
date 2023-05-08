"""
Measure how guesser accuracy globally depends on the words used. More
concretely, for every word in vocabulary estimate probability that guesser
will choose the correct speaker given that word have been selected and other
words are chosen randomly.
"""


from typing import Union
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything
import click
from isr.timit import TimitXVectors, read_words_txt
from isr.nnet import Guesser, Verifier
from isr.simple_agents import RandomAgent


@click.command()
@click.option("-V/--isv", "verification", is_flag=True, default=False,
              help="perform speaker verification instead of default speaker " +
              "recognition")
@click.option("--sd-file-gv", type=click.Path(),
              default="./models/guesser.pth",
              help="path to file with guesser/verifier state_dict")
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="[only ISR] number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--backend", type=click.Choice(["mlp", "cs"]), default="mlp",
              help="[only ISV] backend to use for speaker verification")
@click.option("--num-envs", "--batch-size", type=int, default=200,
              help="number of environments (games) to run in parallel")
@click.option("--episodes", "--test-games", type=int, default=100000,
              help="total number of episodes (games) to run")
def main(verification: bool, sd_file_gv: str, seed: int, split_seed: int,
         num_speakers: int, num_words: int, backend: str, num_envs: int,
         episodes: int):
    seed_everything(seed)
    dset = TimitXVectors(seed=split_seed)
    if verification:
        model = Verifier(emb_dim=dset.emb_dim, backend=backend)
        if sd_file_gv == "./models/guesser.pth":
            sd_file_gv = "./models/verifier.pth"
        model.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))
    else:
        model = Guesser(emb_dim=dset.emb_dim)
        model.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))

    # read file with words
    word_dict = read_words_txt("data/words/WORDS.TXT")
    # word_id -> word (string)
    word_ids = sorted(word_dict.keys())
    V = len(word_dict)

    # evaluate guesser on train and val subsets
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    for subset, ax in zip(["train", "val"], [ax1, ax2]):
        word_acc = evaluate(model, dset, subset, num_speakers, num_words,
                            num_envs, episodes)
        acc_dict = dict(zip(word_ids, word_acc))
        save_scores(acc_dict, f"output/word_scores_{subset}.csv")
        bar_plot(word_acc, ax, ylabel=f"{subset} accuracy")
    ax2.set_xticks(np.arange(V))
    ax2.set_xticklabels([word_dict[wid] for wid in word_ids],
                        rotation="vertical")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig("output/word_scores.png", dpi=75)


def evaluate(model: Union[Guesser, Verifier], dset: TimitXVectors, subset: str,
             num_speakers: int, num_words: int, num_envs: int, episodes: int
             ) -> np.ndarray:
    is_isr = isinstance(model, Guesser)
    # how many times every word was used
    words_used_count = np.zeros(dset.vocab_size, dtype=np.int_)
    # how many times word was used in a successful game
    words_success_count = np.zeros_like(words_used_count)
    # random word sampling / torch.multinomial wrapper
    rand_agent = RandomAgent(total_words=dset.vocab_size)

    # test model on `episodes` games
    sampled_episodes = 0
    model.eval()
    with tqdm(total=episodes) as pbar:
        while sampled_episodes < episodes:
            # sampling + model forward pass
            bs = min(num_envs, episodes - sampled_episodes)
            if is_isr:
                g, speaker_ids, targets = dset.sample_isr_games(
                    bs, subset, num_speakers)
            else:
                g, speaker_ids, targets = dset.sample_isv_games(bs, subset)
            word_inds = rand_agent.sample(bs, num_words)
            x = dset.get_word_embeddings(speaker_ids, word_inds)
            with torch.no_grad():
                output = model.forward(g, x)
                if is_isr:
                    predictions = output.argmax(dim=1)
                else:
                    predictions = output.round().long()
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


def save_scores(data: dict, file: str):
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
