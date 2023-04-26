from typing import Union
from tqdm import tqdm
import numpy as np
import torch
from pytorch_lightning import seed_everything
import click
from isr.timit import TimitXVectors
from isr.nnet import Guesser, Verifier
from isr.simple_agents import HeuristicAgent, RandomAgent


@click.command()
@click.option("-V/--isv", "verification", is_flag=True, default=False,
              help="perform speaker verification instead of default speaker " +
              "recognition")
@click.option("-A/--all", "all_subsets", is_flag=True, default=False)
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("--wscore-file", type=click.Path(),
              default="./models/word_scores.csv", help="file with word scores")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="[only ISR] number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--backend", type=click.Choice(["mlp", "cs"]), default="mlp",
              help="[only ISV] backend to use for speaker verification")
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
def main(verification: bool, all_subsets: bool, seed: int, split_seed: int,
         wscore_file: str, num_speakers: int, num_words: int, backend: str,
         num_envs: int, episodes: int, random_agent: bool,
         agent_num_words: int, nonuniform: bool, temperature: float):
    # load dataset and guesser
    seed_everything(seed)
    dset = TimitXVectors(seed=split_seed)
    if verification:
        model = Verifier(emb_dim=512, backend=backend)
        model.load_state_dict(torch.load("models/verifier.pth",
                                         map_location="cpu"))
    else:
        model = Guesser(emb_dim=512)
        model.load_state_dict(torch.load("models/guesser.pth",
                                         map_location="cpu"))

    # read word scores
    if random_agent:
        agent = RandomAgent(total_words=len(dset.words))
    else:
        word_scores = read_word_scores(wscore_file)
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
        acc = evaluate(model, dset, agent, subset, num_speakers, num_words,
                       num_envs, episodes)
        print(f"Accuracy on {subset}: {acc}")


def read_word_scores(path: str):
    scores = []
    with open(path, "r") as fin:
        for line in fin:
            acc = line.rstrip().split(",")[1]
            scores.append(float(acc))
    return np.array(scores, dtype=float)


def evaluate(model: Union[Guesser, Verifier], dset: TimitXVectors,
             agent: Union[HeuristicAgent, RandomAgent], subset: str,
             num_speakers: int, num_words: int, num_envs: int, episodes: int):
    is_isr = isinstance(model, Guesser)
    sampled_episodes = 0
    accuracy = 0.0
    model.eval()
    with tqdm(total=episodes) as pbar:
        while sampled_episodes < episodes:
            bs = min(num_envs, episodes - sampled_episodes)
            if is_isr:
                g, speaker_ids, targets = dset.sample_isr_games(
                    bs, subset, num_speakers)
            else:
                g, speaker_ids, targets = dset.sample_isv_games(bs, subset)
            word_inds = agent.sample(num_envs, num_words)
            x = dset.get_word_embeddings(speaker_ids, word_inds)
            with torch.no_grad():
                output = model.forward(g, x)
                if is_isr:
                    predictions = output.argmax(dim=1)
                else:
                    predictions = output.round().long()
                acc = (predictions == targets).float().mean().item()
            accuracy += acc * bs / episodes

            sampled_episodes += bs
            pbar.update(bs)

    return accuracy


if __name__ == "__main__":
    main()
