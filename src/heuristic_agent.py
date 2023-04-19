from tqdm import tqdm
import numpy as np
import torch
from pytorch_lightning import seed_everything
import click
from isr.timit import TimitXVectors
from isr.nnet import Guesser
from isr.simple_agents import HeuristicAgent, RandomAgent


@click.command()
@click.option("-A/--all", "all_subsets", is_flag=True, default=False)
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
@click.option("--random-agent", is_flag=True, default=False,
              help="sample words with a random agent")
@click.option("-k", "--agent-num-words", type=int, default=10,
              help="number of top scoring words agent will use")
@click.option("--nonuniform", is_flag=True, default=False,
              help="use higher-scoring words more often")
@click.option("--temperature", type=float, default=0.05,
              help="softmax temperature for converting scores into " +
              "probabilities")
def main(all_subsets: bool, seed: int, split_seed: int, num_speakers: int,
         num_words: int, num_envs: int, episodes: int, random_agent: bool,
         agent_num_words: int, nonuniform: bool, temperature: float):
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
        acc = evaluate(guesser, dset, agent, subset, num_speakers, num_words,
                       num_envs, episodes)
        print(f"Accuracy on {subset}: {acc}")


def read_word_scores(path: str):
    scores = []
    with open(path, "r") as fin:
        for line in fin:
            acc = line.rstrip().split(",")[1]
            scores.append(float(acc))
    return np.array(scores, dtype=float)


def evaluate(guesser, dset, agent, subset, num_speakers, num_words, num_envs,
             episodes):
    sampled_episodes = 0
    accuracy = 0.0
    guesser.eval()
    with tqdm(total=episodes) as pbar:
        while sampled_episodes < episodes:
            bs = min(num_envs, episodes - sampled_episodes)
            g, target_ids, targets = dset.sample_games(
                batch_size=bs,
                subset=subset,
                num_speakers=num_speakers
            )
            word_inds = agent.sample(num_envs, num_words)
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
