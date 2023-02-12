from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim, Tensor
from common import PathLike
from nnet import Guesser
import timit


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SPEAKERS = 5
NUM_WORDS = 3
EPOCHS = 10
ITERATIONS = 100
BATCH_SIZE = 32


def main():
    dset = TimitXVectors()
    guesser = Guesser(output_format="logprob").to(DEVICE)
    optimizer = optim.Adam(guesser.parameters())

    train_results = np.zeros((2, EPOCHS))
    val_results = np.zeros_like(train_results)
    for epoch in range(EPOCHS):
        train_results[:, epoch] = run_n_iterations(
            dset, guesser, optimizer, "train", BATCH_SIZE, ITERATIONS)
        print(f"[train] loss = {train_results[0, epoch]}, " +
              f"accuracy = {train_results[1, epoch]}")
        val_results[:, epoch] = run_n_iterations(
            dset, guesser, optimizer, "val", BATCH_SIZE, ITERATIONS // 4)
        print(f"[validation] loss = {val_results[0, epoch]}, " +
              f"accuracy = {val_results[1, epoch]}")

    test_loss, test_acc = run_n_iterations(
        dset, guesser, optimizer, "test", BATCH_SIZE, 1000)
    print(f"[test] loss = {test_loss}, accuracy = {test_acc}")

    plt.figure()
    epochs = np.arange(1, EPOCHS + 1)
    plt.plot(epochs, train_results[0], "b-.", label="train")
    plt.plot(epochs, val_results[0], "r-.", label="validation")
    plt.legend(loc="center right")
    plt.twinx()
    plt.plot(epochs, train_results[1], "b:.", label="train")
    plt.plot(epochs, val_results[1], "r:.", label="validation")
    plt.show()


class TimitXVectors():
    def __init__(self, data_dir: PathLike = "./data", val_size: float = 0.2):
        super().__init__()
        data_dir = Path(data_dir)

        # read TIMIT info
        doc_dir = data_dir / "TIMIT/DOC"
        # prompt_id -> prompt
        self.prompts = timit.read_prompts(doc_dir / "PROMPTS.TXT")
        # speaker_id -> (prompt_ids)
        self.spkrsent = timit.read_spkrsent(doc_dir / "SPKRSENT.TXT")
        # dataframe with speaker info
        self.spkrinfo = timit.read_spkrinfo(doc_dir / "SPKRINFO.TXT")

        # split speakers into subsets
        self.speakers = {"train": [], "val": [], "test": []}
        for spkr_id, spkr_info in self.spkrinfo.iterrows():
            if spkr_info["Use"] == "TRN":
                if np.random.random() > val_size:
                    self.speakers["train"].append(spkr_id)
                else:
                    self.speakers["val"].append(spkr_id)
            else:
                assert spkr_info["Use"] == "TST", "SPKRINFO.TXT read error"
                self.speakers["test"].append(spkr_id)

        # load voice prints (speaker xvectors)
        self.xvectors = {}
        self.xvectors["train"] = \
            np.load(data_dir / "xvectors_train/spk_xvector.npz")
        self.xvectors["test"] = \
            np.load(data_dir / "xvectors_test/spk_xvector.npz")

        # word x-vectors
        self.words = {}
        with open(data_dir / "words/WORDS.TXT", "r") as fin:
            for line in fin:
                word, word_id = line.rstrip().split()
                self.words[word_id] = word
        self.word_ids = sorted(self.words.keys())
        self.xvectors["words"] = \
            np.load(data_dir / "xvectors_words/xvector.npz")

    def sample(self, subset: str = "train", k: int = 5, t: int = 3
               ) -> Tuple[Tensor, Tensor, int]:
        # sample k speakers
        spkr_inds = np.random.choice(np.arange(len(self.speakers[subset])),
                                     size=k, replace=False)
        spkrs = np.array([self.speakers[subset][ind] for ind in spkr_inds])

        # get speaker voice prints
        xv_subset = "train" if subset == "val" else subset
        voice_prints = np.stack([self.xvectors[xv_subset][spkr]
                                 for spkr in spkrs], axis=0)

        # select target speaker
        target = np.random.randint(k)
        target_spkr = spkrs[target]

        # sample words
        words = np.random.choice(list(self.words.keys()), size=t,
                                 replace=False)
        word_emb = np.stack([
            self.xvectors["words"].get(f"{target_spkr}_{word}", np.zeros(128))
            for word in words])

        return (torch.FloatTensor(word_emb),
                torch.FloatTensor(voice_prints),
                torch.tensor(np.array(target, dtype=np.int64)))

    def get_batch(self, bs: int = 32, subset: str = "train",
                  k: int = 5, t: int = 3):
        samples = [self.sample(subset, k, t) for _ in range(bs)]
        return map(torch.stack, zip(*samples))


def forward_step(guesser: Guesser, batch: Tuple):
    X, G, target = [t.to(DEVICE) for t in batch]
    batch_size = len(target)
    output = guesser(X, G)
    accuracy = float((output.argmax(-1) == target).sum() / batch_size)
    if isinstance(guesser.f_out, nn.Identity):
        loss_fn = nn.CrossEntropyLoss()
    elif isinstance(guesser.f_out, nn.LogSoftmax):
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


def run_n_iterations(data: TimitXVectors, guesser: Guesser,
                     optimizer: optim.Optimizer, mode: str, batch_size: int,
                     iterations: int) -> Tuple[float, float]:
    avg_loss, avg_accuracy = 0., 0.
    for _ in range(iterations):
        batch = data.get_batch(batch_size, subset=mode,
                               k=NUM_SPEAKERS, t=NUM_WORDS)
        if mode == "train":
            loss, acc = training_step(guesser, batch, optimizer)
        else:
            loss, acc = evaluation_step(guesser, batch)
        avg_loss += loss / iterations
        avg_accuracy += acc / iterations
    return avg_loss, avg_accuracy


if __name__ == "__main__":
    main()
