import torch
from torch import nn
from torch.nn import functional as F
from deep_learning.utils.module import ExtendedModule


class SoftmaxRegression(ExtendedModule):
    def __init__(self, num_outputs, lr) -> None:
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y, averaged=True):
        return F.cross_entropy(y_hat, y, reduction="mean" if averaged else "none")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
