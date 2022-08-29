import torch
from random import shuffle


class SynteticRegressionData:
    def __init__(self, w, b, noise=0.01, num_train=1_000, num_val=1_0000, batch_size=32):
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size

        # Total number of observations(rows) to generate
        n = self.num_train + self.num_val

        # Input data
        self.x = torch.randn(n, len(w))
        self.noise = torch.randn(n, 1) * noise

        # input matrix * transposed(column vector) weights + bayesian + noise
        self.y = torch.matmul(self.x, w.reshape((-1, 1))) + b + self.noise

    def _get_dataloader(self, is_train):
        # Ilustrating generally data loading, no prefetch, in ram data....
        if is_train:
            indices = list(range(0, self.num_train))
            shuffle(indices)
        else:
            indices = list(
                range(self.num_train, self.num_train + self.num_val))

        for i in range(0, len(indices), self.batch_size):
            batch_indicies = torch.tensor(indices[i: i+self.batch_size])
            yield self.x[batch_indicies], self.y[batch_indicies]

    def _get_tensor_loader(self, tensors, is_train, indicies=slice(0, None)):
        tensors = tuple(a[indicies] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=is_train)

    def get_dataloader(self, is_train):
        # Slightly concise implementation of data representation, could also use torch.utils.data.Dataset
        indices = slice(0, self.num_train) if is_train else slice(self.num_train, None)
        return self._get_tensor_loader((self.x, self.y), is_train, indices)

    def train_dataloader(self):
        return self.get_dataloader(is_train=True)
    
    def val_dataloader(self):
        return self.get_dataloader(is_train=False)
