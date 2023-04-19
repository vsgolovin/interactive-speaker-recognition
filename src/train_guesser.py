"""
Notes:
  * Using not 128-, but 512-dimensional embeddings, as it is not clear how
   (and why) to perform dimensionality reduction.
  * Paper states, that guesser was trained with batch size of 1024. Not sure
    if this means 1024 games or 1024 speakers. It also says that there were
    45k training games in total, which would mean just 45 1024-game batches.
  * Final test accuracy higher than in the paper.
"""

from pathlib import Path
from typing import Generator, Optional, Union
import click
import torch
from torch import nn, optim, Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from isr.nnet import Guesser
from isr import timit


@click.group()
def cli():
    pass


@cli.command()
@click.option("--sd-file", type=click.Path(), default="./output/guesser.pth",
              help="file to save guesser state_dict to")
@click.option("--seed", type=int, default=2303, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--batch-size", type=int, default=100, help="batch size")
@click.option("--iterations", type=int, default=200,
              help="number of iterations per epoch")
@click.option("--max-epochs", type=int, default=50,
              help="maximum number of epochs (training uses EarlyStopping)")
@click.option("--lr", type=float, default=1e-4, help="learning rate")
@click.option("--weight-decay", type=float, default=1e-4,
              help="L2 regularization")
def train(sd_file: str, seed: int, split_seed: int, num_speakers: int,
          num_words: int, batch_size: int, iterations: int, max_epochs: int,
          lr: float, weight_decay: float):
    pl.seed_everything(seed)
    dm = XVectorsForGuesser(num_speakers, num_words, batch_size, iterations,
                            seed=split_seed)
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
    model = LitGuesser.load_from_checkpoint(save_best.best_model_path)
    model.save(sd_file)


@cli.command()
@click.option("-A/--all", "all_subsets", is_flag=True, default=False)
@click.option("--sd-file", type=click.Path(), default="./output/guesser.pth",
              help="path to file with guesser state_dict")
@click.option("--seed", type=int, default=2303, help="global seed")
@click.option("--split-seed", type=int, default=42,
              help="seed used to perform train-val split")
@click.option("-K", "--num-speakers", type=int, default=5,
              help="number of speakers present in every game")
@click.option("-T", "--num-words", type=int, default=3,
              help="number of words asked in every game")
@click.option("--batch-size", type=int, default=100, help="batch size")
@click.option("--test-games", type=int, default=20000,
              help="total number of games (batch_size * iterations)")
def test(all_subsets: bool, sd_file: str, seed: int, split_seed: int,
         num_speakers: int, num_words: int, batch_size: int, test_games: int):
    pl.seed_everything(seed)
    iterations = test_games // batch_size
    dm = XVectorsForGuesser(num_speakers, num_words, batch_size, iterations,
                            seed=split_seed)
    guesser = LitGuesser()
    guesser.model.load_state_dict(torch.load(sd_file,
                                             map_location="cpu"))
    trainer = pl.Trainer(logger=False, accelerator="cpu")
    dloaders = [dm.test_dataloader()]
    if all_subsets:
        dloaders = [dm.train_dataloader(), dm.val_dataloader()] + dloaders
    trainer.test(model=guesser, dataloaders=dloaders)


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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc = self._forward_pass(batch)
        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())

    def save(self, f: Union[Path, str]):
        "Save only Guesser state_dict"
        torch.save(self.model.state_dict(), f)

    def load(self, f: Union[Path, str]):
        "Loader Guesser state_dict from file"
        self.model.load_state_dict(torch.load(f))


class XVectorsForGuesser(pl.LightningDataModule):
    def __init__(self, num_speakers: int, num_words: int,
                 batch_size: int, iterations_per_epoch: int,
                 data_dir: Union[Path, str] = "./data", val_size: float = 0.2,
                 seed: Optional[int] = None):
        super().__init__()
        self.K = num_speakers
        self.T = num_words
        self.dset = timit.TimitXVectors(data_dir, val_size, seed)
        self.batch_size = batch_size
        self.iterations = iterations_per_epoch

    def _dataloader(self, subset: str) -> Generator:
        count = 0
        while count < self.iterations:
            g, target_ids, targets = self.dset.sample_games(
                batch_size=self.batch_size,
                subset=subset,
                num_speakers=self.K
            )
            x = self.dset.sample_words(target_ids, self.T)
            yield g, x, targets
            count += 1

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")


if __name__ == "__main__":
    cli.add_command(train)
    cli.add_command(test)
    cli()
