import torch
from torch.nn import Module
from deep_learning.utils.module import ExtendedModule


class LinearRegression(ExtendedModule):
    def __init__(self, num_inputs, lr, sigma=0.01) -> None:
        super().__init__()
        self.lr = lr
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('loss', l, train=False)



class SGD:
    def __init__(self, params, lr) -> None:
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()