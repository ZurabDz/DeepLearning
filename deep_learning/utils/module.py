"""
Extending pytorch.nn.Module for additional functionality(non concise, later will be added better)
"""
import torch
from deep_learning.utils.progress_board import ProgressBoard


class ExtendedModule(torch.nn.Module):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1) -> None:
        super().__init__()
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    def plot(self, key, value, train):
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(
            x,
            value.detach().cpu().numpy(),
            ("train_" if train else "val_") + key,
            every_n=int(n),
        )

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", l, train=False)
