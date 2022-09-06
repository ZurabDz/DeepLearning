from deep_learning.utils.module import ExtendedModule
import torch


class LinearRegression(ExtendedModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        self.net = torch.nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = torch.nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)