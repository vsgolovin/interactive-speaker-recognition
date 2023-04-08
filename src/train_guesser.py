"""
Notes:
  * Using not 128-, but 512-dimensional embeddings, as it is not clear how to
    perform dimensionality reduction.
  * Paper states, that guesser was trained with batch size of 1024. Not sure
    if this means 1024 games or 1024 speakers. It also says that there were
    45k training games in total, which would mean just 45 1024-game batches.
  * Final test accuracy appears to be higher than in the paper.
"""

from typing import Generator, Optional
import torch
from torch import nn, optim, Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from isr.common import PathLike
from isr.nnet import Guesser
from isr.data import timit


NUM_SPEAKERS = 5
NUM_WORDS = 3


def main(split_seed: int, batch_size: int, iterations: int,
         max_epochs: int, lr: float, weight_decay: float):
    dm = XVectorsForGuesser(batch_size, iterations, seed=split_seed)
    guesser = LitGuesser(lr, weight_decay)
    early_stopping = EarlyStopping(monitor="val_acc", patience=5, mode="max")
    save_best = ModelCheckpoint(save_top_k=1, filename="{epoch}-{val_acc:.2f}")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger = pl.loggers.TensorBoardLogger(save_dir="./output",
                                          name="guesser_logs")
    trainer = pl.Trainer(logger=logger, accelerator=accelerator,
                         max_epochs=max_epochs,
                         callbacks=[early_stopping, save_best],
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(model=guesser, datamodule=dm)


class LitGuesser(pl.LightningModule):
    def __init__(self, lr: float = 3e-4, weight_decay: float = 0.):
        super().__init__()
        self.model = Guesser(output_format="logit")
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        return self.model(g, x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)

    def _forward_pass(self, batch):
        g, x, target = batch
        output = self.model(g, x)
        loss = self.loss_fn(output, target)
        acc = (output.detach().argmax(dim=1) == target).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._forward_pass(batch)
        self.log("train_loss", loss.item(), prog_bar=False)
        self.log("train_acc", acc.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._forward_pass(batch)
        self.log("val_loss", loss.item(), prog_bar=False)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._forward_pass(batch)
        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())


class XVectorsForGuesser(pl.LightningDataModule):
    def __init__(self, batch_size: int, iterations_per_epoch: int,
                 data_dir: PathLike = "./data", val_size: float = 0.2,
                 seed: Optional[int] = None):
        super().__init__()
        self.dset = timit.TimitXVectors(data_dir, val_size, seed)
        self.batch_size = batch_size
        self.iterations = iterations_per_epoch

    def _dataloader(self, subset: str) -> Generator:
        count = 0
        while count < self.iterations:
            g, target_ids, targets = self.dset.sample_games(
                batch_size=self.batch_size,
                subset=subset,
                num_speakers=NUM_SPEAKERS
            )
            x = self.dset.sample_words(target_ids, NUM_WORDS)
            yield g, x, targets
            count += 1

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=42,
                        help="seed used to perform train-val split")
    parser.add_argument("--bs", type=int, default=100, help="batch size")
    parser.add_argument("--iterations", type=int, default=200,
                        help="number of iterations per epoch")
    parser.add_argument("--max-epochs", type=int, default=50,
                        help="maximum number of epochs " +
                        "(training uses EarlyStopping)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization")
    args = parser.parse_args()

    main(
        split_seed=args.split_seed,
        batch_size=args.bs,
        iterations=args.iterations,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
