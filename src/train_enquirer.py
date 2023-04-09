from pathlib import Path
from typing import Optional
from tqdm import tqdm
import click
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import seed_everything
from isr.nnet import Enquirer, Guesser
from isr.envtools import IsrEnvironment
from isr.data import timit
from isr.ppo import Buffer, PPO


NUM_WORDS = 3
NUM_SPEAKERS = 5


@click.command()
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("--num-envs", type=int, default=33,
              help="number of ISR environments (games) to run in parallel")
@click.option("--episodes-per-update", type=int, default=330,
              help="number of episodes to sample before performing an update")
@click.option("--eval-period", type=int, default=60,
              help="number of updates to perform before every evaluation")
@click.option("--num-updates", type=int, default=300,
              help="total number of model updates to perform")
@click.option("--batch-size", type=int, default=200, help="batch size")
@click.option("--epochs-per-update", type=int, default=2,
              help="times to iterate over collected data on every update")
@click.option("--lr-actor", type=float, default=1e-4,
              help="PPO actor learning rate")
@click.option("--lr-critic", type=float, default=1e-4,
              help="PPO critic learning rate")
@click.option("--ppo-clip", type=float, default=0.2,
              help="PPO clipping parameter (epsilon)")
@click.option("--grad-clip", type=float, required=False,
              help="PPO gradient clipping")
@click.option("--entropy", type=float, required=False,
              help="PPO entropy penalty coefficient")
def main(seed: int, split_seed: int, num_envs: int, episodes_per_update: int,
         eval_period: int, num_updates: int, batch_size: int,
         epochs_per_update: int, lr_actor: float, lr_critic: float,
         ppo_clip: float, grad_clip: Optional[float],
         entropy: Optional[float]):
    seed_everything(seed)
    output_dir = Path("output")
    dset = timit.TimitXVectors(seed=split_seed)
    guesser = Guesser(emb_dim=512)
    guesser.load_state_dict(torch.load("models/guesser.pth",
                                       map_location="cpu"))
    env = IsrEnvironment(dset, guesser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = PPO(512, len(dset.words), device=device, lr_actor=lr_actor,
              lr_critic=lr_critic, ppo_clip=ppo_clip, grad_clip=grad_clip,
              entropy=entropy)
    buffer = Buffer(num_words=NUM_WORDS)

    # train enquirer
    avg_rewards = np.zeros(num_updates // eval_period)
    ppo.train()
    BATCHES_PER_UPDATE = episodes_per_update // num_envs
    NUM_EPISODES = episodes_per_update * num_updates
    with tqdm(desc="PPO training", total=NUM_EPISODES) as pbar:
        for i in range(num_updates):
            # actual training
            for _ in range(BATCHES_PER_UPDATE):
                states = env.reset("train", batch_size=num_envs,
                                   num_speakers=NUM_SPEAKERS,
                                   num_words=NUM_WORDS)
                for _ in range(NUM_WORDS):
                    actions, probs, values = ppo.step(states)
                    new_states, rewards = env.step(actions)
                    buffer.append(states, actions, probs, rewards, values)
                    states = new_states
                pbar.update(num_envs)
            ppo.update(buffer, batch_size, epochs_per_update)
            buffer.empty()

            # evaluate on validation set
            if (i + 1) % eval_period == 0:
                r_avg = evaluate(ppo, env, "val", progress_bar=False)
                pbar.set_postfix({"r_avg": r_avg})
                avg_rewards[i // eval_period] = r_avg
                ppo.train()

    # save actor and critic weights
    ppo.save(output_dir)

    # plot avg. reward on validation set
    eval_step = episodes_per_update * eval_period
    episode_count = np.arange(eval_step, NUM_EPISODES + 1, eval_step)
    plt.figure()
    plt.plot(episode_count, avg_rewards, "bo-")
    plt.ylabel("Avg. reward on validation set")
    plt.xlabel("Episodes")
    plt.savefig(output_dir / "enquirer_training.png", dpi=75)


def evaluate(ppo: Enquirer, env: IsrEnvironment,
             subset: str = "val", episodes: int = 20000,
             parallel_envs: int = 50, progress_bar: bool = True) -> float:
    ppo.eval()
    all_rewards = np.zeros(episodes)

    if progress_bar:
        pbar = tqdm(desc=f"SR evaluation on {subset}", total=episodes)

    performed = 0
    while performed < episodes:
        cur_envs = min(episodes - performed, parallel_envs)
        states = env.reset(subset, batch_size=cur_envs,
                           num_speakers=NUM_SPEAKERS, num_words=NUM_WORDS)
        for _ in range(NUM_WORDS):
            actions, _, _ = ppo.step(states)
            new_states, rewards = env.step(actions)
            states = new_states
        all_rewards[performed:performed + cur_envs] = \
            rewards.cpu().numpy()
        performed += cur_envs

        if progress_bar:
            pbar.update(cur_envs)

    if progress_bar:
        pbar.close()

    return np.mean(all_rewards)


if __name__ == "__main__":
    main()
