from pathlib import Path
import re
from typing import Union
from tqdm import tqdm
import click
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything
from isr.nnet import Guesser, Verifier, Enquirer, CodebookEnquirer
from isr.envtools import IsrEnvironment, IsvEnvironment
from isr import timit
from isr.ppo import Buffer, PPO


@click.group()
def cli():
    pass


@cli.command()
@click.option("-C/--codebook", "use_codebook", is_flag=True, default=False,
              help="use CodebookEnquirer instead of regular Enquirer")
@click.option("-V/--isv", "verification", is_flag=True, default=False,
              help="perform speaker verification instead of default speaker " +
              "recognition")
@click.option("--sd-file-gv", type=click.Path(),
              default="./models/guesser.pth",
              help="path to file with guesser/verifier state_dict")
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-N", "--noise", is_flag=True, default=False,
              help="whether to use noisy word recordings")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--backend", type=click.Choice(["mlp", "cs"]), default="mlp",
              help="[ISV only] backend to use for speaker verification")
@click.option("--num-envs", type=int, default=33,
              help="number of ISR environments (games) to run in parallel")
@click.option("--episodes-per-update", type=int, default=330,
              help="number of episodes to sample before performing an update")
@click.option("--eval-period", type=int, default=60,
              help="number of updates to perform before every evaluation")
@click.option("--num-updates", type=int, default=600,
              help="total number of model updates to perform")
@click.option("--batch-size", type=int, default=500, help="batch size")
@click.option("--epochs-per-update", type=int, default=2,
              help="times to iterate over collected data on every update")
@click.option("--lr-actor", type=float, default=2e-5,
              help="PPO actor learning rate")
@click.option("--lr-critic", type=float, default=2e-5,
              help="PPO critic learning rate")
@click.option("--ppo-clip", type=float, default=0.2,
              help="PPO clipping parameter (epsilon)")
@click.option("--entropy", type=float, default=0.01,
              help="PPO entropy penalty coefficient")
@click.option("--grad-clip", type=float, default=1.0,
              help="PPO gradient clipping")
def train(use_codebook: bool, verification: bool, sd_file_gv: str, seed: int,
          split_seed: int, noise: bool, num_speakers: int, num_words: int,
          backend: str, num_envs: int, episodes_per_update: int,
          eval_period: int, num_updates: int, batch_size: int,
          epochs_per_update: int, lr_actor: float, lr_critic: float,
          ppo_clip: float, entropy: float, grad_clip: float):
    seed_everything(seed)
    hparams = locals()
    output_dir = Path("output")
    dset = timit.TimitXVectors(seed=split_seed, noisy_words=noise)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_codebook:
        codebook = dset.create_codebook("train")
        word_inds = torch.arange(0, dset.vocab_size, 2)  # [0, 2, ..., 18]
        enquirer = CodebookEnquirer(len(word_inds), dset.emb_dim)
        enquirer.load_codebook(codebook[word_inds], update_stats=True)
    else:
        enquirer = Enquirer(emb_dim=dset.emb_dim, n_outputs=dset.vocab_size)
        word_inds = None
    ppo = PPO(enquirer, dset.emb_dim, device=device, lr_actor=lr_actor,
              lr_critic=lr_critic, ppo_clip=ppo_clip, entropy=entropy,
              grad_clip=None if grad_clip == 0 else grad_clip)
    buffer = Buffer(num_words=num_words)
    if verification:
        verifier = Verifier(emb_dim=dset.emb_dim, backend=backend)
        if sd_file_gv == "./models/guesser.pth":
            sd_file_gv = "./models/verifier.pth"
        verifier.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))
        env = IsvEnvironment(dset, verifier, word_inds)
    else:
        guesser = Guesser(emb_dim=dset.emb_dim)
        guesser.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))
        env = IsrEnvironment(dset, guesser, word_inds)

    # tensorboard logger
    log_dir = create_log_dir()
    writer = SummaryWriter(log_dir=log_dir)

    # train enquirer
    avg_rewards = np.zeros(num_updates // eval_period)
    ppo.train()
    BATCHES_PER_UPDATE = episodes_per_update // num_envs
    NUM_EPISODES = episodes_per_update * num_updates
    episode_count = 0
    max_reward = 0.0
    state_reset_kwargs = {"subset": "train", "batch_size": num_envs,
                          "num_words": num_words}
    if not verification:
        state_reset_kwargs["num_speakers"] = num_speakers
    with tqdm(desc="PPO training", total=NUM_EPISODES) as pbar:
        for i in range(num_updates):
            # actual training
            for _ in range(BATCHES_PER_UPDATE):
                states = env.reset(**state_reset_kwargs)
                for _ in range(num_words):
                    actions, probs, values = ppo.step(states)
                    new_states, rewards = env.step(actions)
                    buffer.append(states, actions, probs, rewards, values)
                    states = new_states
                pbar.update(num_envs)
                episode_count += num_envs
                writer.add_scalar("reward/train", rewards.mean().item(),
                                  global_step=episode_count)
                if use_codebook:
                    writer.add_scalar(
                        "temperature",
                        ppo.actor.model.t_coeff.exp().item(),
                        global_step=episode_count
                    )
            losses = ppo.update(buffer, batch_size, epochs_per_update)
            buffer.empty()
            for k, v in losses.items():
                writer.add_scalar(f"loss/{k}", v, global_step=episode_count)

            # evaluate on validation set
            if (i + 1) % eval_period == 0:
                r_avg = evaluate(ppo, env, "val", num_speakers=num_speakers,
                                 num_words=num_words, progress_bar=False)
                if r_avg > max_reward:
                    max_reward = r_avg
                    ppo.save(output_dir)
                pbar.set_postfix({"r_avg": r_avg})
                avg_rewards[i // eval_period] = r_avg
                writer.add_scalar("reward/val", r_avg,
                                  global_step=episode_count)
                ppo.train()

    # save hyperparams and best reward (accuracy)
    # run_name=log_dir => do not create subfolder
    writer.add_hparams(hparams, {"val_acc": max_reward},
                       run_name=str(log_dir.absolute()))

    # plot avg. reward on validation set
    eval_step = episodes_per_update * eval_period
    episode_count = np.arange(eval_step, NUM_EPISODES + 1, eval_step)
    plt.figure()
    plt.plot(episode_count, avg_rewards, "bo-")
    plt.ylabel("Avg. reward on validation set")
    plt.xlabel("Episodes")
    plt.savefig(output_dir / "enquirer_training.png", dpi=75)


@cli.command()
@click.option("-C/--codebook", "use_codebook", is_flag=True, default=False,
              help="use CodebookEnquirer instead of regular Enquirer")
@click.option("-V/--isv", "verification", is_flag=True, default=False,
              help="perform speaker verification instead of default speaker " +
              "recognition")
@click.option("-A/--all", "all_subsets", is_flag=True, default=False)
@click.option("--sd-file", type=click.Path(), default="./output/actor.pth",
              help="path to file with enquirer state_dict")
@click.option("--sd-file-gv", type=click.Path(),
              default="./models/guesser.pth",
              help="path to file with guesser/verifier state_dict")
@click.option("--seed", type=int, default=2008, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-N", "--noise", is_flag=True, default=False,
              help="whether to use noisy word recordings")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--backend", type=click.Choice(["mlp", "cs"]), default="mlp",
              help="[ISV only] backend to use for speaker verification")
@click.option("--num-envs", "--batch-size", type=int, default=200,
              help="number of ISR environments (games) to run in parallel")
@click.option("--episodes", "--test-games", type=int, default=20000,
              help="total number of episodes (games) to run")
def test(use_codebook: bool, verification: bool, all_subsets: bool,
         sd_file: str, sd_file_gv: str, seed: int, split_seed: int,
         noise: bool, num_speakers: int, num_words: int, backend: str,
         num_envs: int, episodes: int):
    seed_everything(seed)
    dset = timit.TimitXVectors(seed=split_seed, noisy_words=noise)
    if use_codebook:
        codebook = dset.create_codebook("train")
        word_inds = torch.arange(1, dset.vocab_size, 2)  # [1, 3, ..., 19]
        enquirer = CodebookEnquirer(len(word_inds), dset.emb_dim)
        enquirer.load_state_dict(torch.load(sd_file, map_location="cpu"))
        enquirer.load_codebook(codebook[word_inds], update_stats=False)
    else:
        enquirer = Enquirer(512, len(dset.words))
        enquirer.load_state_dict(torch.load(sd_file, map_location="cpu"))
        word_inds = None
    ppo = PPO(enquirer, dset.emb_dim, device=torch.device("cpu"))

    if verification:
        verifier = Verifier(emb_dim=dset.emb_dim, backend=backend)
        if sd_file_gv == "./models/guesser.pth":
            sd_file_gv = "./models/verifier.pth"
        verifier.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))
        env = IsvEnvironment(dset, verifier, word_inds)
    else:
        guesser = Guesser(emb_dim=dset.emb_dim)
        guesser.load_state_dict(torch.load(sd_file_gv, map_location="cpu"))
        env = IsrEnvironment(dset, guesser, word_inds)

    subsets = ["test"]
    if all_subsets:
        subsets = ["train", "val"] + subsets
    for subset in subsets:
        r = evaluate(ppo, env, subset, num_speakers, num_words,
                     episodes, num_envs, True)
        print(f"Average reward (accuracy) on {subset}: {r}")


def evaluate(ppo: PPO, env: Union[IsrEnvironment, IsvEnvironment],
             subset: str = "val", num_speakers: int = 5, num_words: int = 3,
             episodes: int = 20000, parallel_envs: int = 50,
             progress_bar: bool = True) -> float:
    is_isv = isinstance(env, IsvEnvironment)
    ppo.eval()
    all_rewards = np.zeros(episodes)

    if progress_bar:
        mode = "SV" if is_isv else "SR"
        pbar = tqdm(desc=f"{mode} evaluation on {subset}", total=episodes)

    performed = 0
    state_reset_kwargs = {"subset": subset, "batch_size": parallel_envs,
                          "num_words": num_words}
    if not is_isv:
        state_reset_kwargs["num_speakers"] = num_speakers
    while performed < episodes:
        cur_envs = min(episodes - performed, parallel_envs)
        states = env.reset(**state_reset_kwargs)
        past_actions = torch.zeros((states.size(0), num_words),
                                   dtype=torch.int64)
        for i in range(num_words):
            actions, _, _ = ppo.step(states, past_actions[:, :i])
            new_states, rewards = env.step(actions)
            states = new_states
            past_actions[:, i] = actions
        all_rewards[performed:performed + cur_envs] = \
            rewards.cpu().numpy()
        performed += cur_envs

        if progress_bar:
            pbar.update(cur_envs)

    if progress_bar:
        pbar.close()

    return np.mean(all_rewards)


def create_log_dir(root="output/enquirer_logs"):
    root = Path(root)
    idx = -1
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            m = re.match(r"version_([\d]+)", child.stem)
            if m:
                idx = max(idx, int(m.group(1)))
    log_dir = root / f"version_{idx + 1}"
    log_dir.mkdir(parents=True)
    return log_dir


if __name__ == "__main__":
    cli()
