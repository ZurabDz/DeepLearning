"""
Extending pytorch.nn.Module for additional functionality(non concise, later will be added better)
"""
import torch

class ProgressBoard:
    pass

class Module(torch.nn.Module):
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
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self):
        pass

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)
     