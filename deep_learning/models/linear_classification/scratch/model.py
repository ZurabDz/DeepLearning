import torch
from deep_learning.optimizers.sgd import SGD
from deep_learning.utils.module import ExtendedModule

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]).mean()


class SoftmaxRegression(ExtendedModule):
    def __init__(self, num_inputs, num_outpus, lr, sigma=0.1):
        super().__init__()
        self.lr = lr
        self.w = torch.normal(0, sigma, size=(num_inputs, num_outpus), requires_grad=True)
        self.b = torch.zeros(num_outpus, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def forward(self, x):
        return softmax(torch.matmul(x.reshape((-1, self.w.shape[0])), self.w) + self.b)

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])