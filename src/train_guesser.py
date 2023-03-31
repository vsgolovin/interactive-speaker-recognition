"""
Notes:
  * Using not 128-, but 512-dimensional embeddings, as it is not clear how to
    perform dimensionality reduction.
  * Paper states, that guesser was trained with batch size of 1024. Not sure
    if this means 1024 games or 1024 speakers. It also says that there were
    45k training games in total, which would mean just 45 1024-game batches.
  * Final test accuracy appears to be higher than in the paper.
"""

from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from nnet import Guesser
import timit


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SPEAKERS = 5
NUM_WORDS = 3
EPOCHS = 40
ITERATIONS = 200  # per epoch
BATCH_SIZE = 50
TEST_GAMES = 20000
SPLIT_SEED = 42


def main():
    # directory to save model weights and training plot
    output_dir = Path("./output")
    if not output_dir.exists():
        output_dir.mkdir()
    else:
        assert output_dir.is_dir()

    # initialize dataset wrapper, model and optimizer
    dset = timit.TimitXVectors(seed=SPLIT_SEED)
    guesser = Guesser(emb_dim=512, output_format="logit").to(DEVICE)
    optimizer = optim.Adam(guesser.parameters())

    # train the model
    train_results = np.zeros((2, EPOCHS))
    val_results = np.zeros_like(train_results)
    for epoch in range(EPOCHS):
        train_results[:, epoch] = run_n_iterations(
            dset, guesser, optimizer, "train", BATCH_SIZE, ITERATIONS)
        print(f"[train] loss = {train_results[0, epoch]}, " +
              f"accuracy = {train_results[1, epoch]}")
        val_results[:, epoch] = run_n_iterations(
            dset, guesser, optimizer, "val", BATCH_SIZE, ITERATIONS // 2)
        print(f"[validation] loss = {val_results[0, epoch]}, " +
              f"accuracy = {val_results[1, epoch]}")

    # check model performance on test set
    test_loss, test_acc = run_n_iterations(
        dset, guesser, optimizer, "test", BATCH_SIZE, TEST_GAMES // BATCH_SIZE)
    print(f"[test] loss = {test_loss}, accuracy = {test_acc}")

    # export model weights
    torch.save(guesser.state_dict(), output_dir / "guesser.pth")

    # plot results
    plt.figure()
    epochs = np.arange(1, EPOCHS + 1)
    plt.plot(epochs, train_results[0], "b.-", label="train")
    plt.plot(epochs, val_results[0], "r.-", label="validation")
    plt.plot([epochs[-1]], [test_loss], "k.", label="test")
    plt.legend(loc="center right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.twinx()
    plt.plot(epochs, train_results[1], "b.:", label="train")
    plt.plot(epochs, val_results[1], "r.:", label="validation")
    plt.plot([epochs[-1]], [test_acc], "k.", label="test")
    plt.ylabel("Accuracy")
    plt.savefig(output_dir / "guesser_training.png", dpi=150)


def forward_step(guesser: Guesser, batch: Tuple):
    G, X, target = [t.to(DEVICE) for t in batch]
    batch_size = len(target)
    output = guesser(G, X)
    accuracy = float((output.argmax(-1) == target).sum() / batch_size)
    if guesser.output_format == "logit":
        loss_fn = nn.CrossEntropyLoss()
    elif guesser.output_format == "logprob":
        loss_fn = nn.NLLLoss()
    else:
        raise Exception("Please use Guesser with 'logit' or 'logprob' output")
    loss = loss_fn(output, target)
    return loss, accuracy


def training_step(guesser: Guesser, batch: Tuple, optimizer: optim.Optimizer):
    guesser.train()
    loss, accuracy = forward_step(guesser, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy


def evaluation_step(guesser: Guesser, batch: Tuple):
    guesser.eval()
    with torch.no_grad():
        loss, accuracy = forward_step(guesser, batch)
    return loss.item(), accuracy


def run_n_iterations(data: timit.TimitXVectors, guesser: Guesser,
                     optimizer: optim.Optimizer, subset: str, batch_size: int,
                     iterations: int) -> Tuple[float, float]:
    avg_loss, avg_accuracy = 0., 0.
    for _ in range(iterations):
        g, target_ids, targets = data.sample_games(batch_size, subset,
                                                   NUM_SPEAKERS)
        x = data.sample_words(target_ids, NUM_WORDS)
        batch = (g, x, targets)
        if subset == "train":
            loss, acc = training_step(guesser, batch, optimizer)
        else:
            loss, acc = evaluation_step(guesser, batch)
        avg_loss += loss / iterations
        avg_accuracy += acc / iterations
    return avg_loss, avg_accuracy


if __name__ == "__main__":
    main()
