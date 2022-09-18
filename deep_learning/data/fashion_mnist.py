from random import shuffle
import torch
import torchvision
from torchvision import transforms
from deep_learning.utils.image import show_images

class FashionMNIST:
    def __call__(self, root, num_workers, batch_size=64, resize=(28, 28)):
        self.trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=self.trans, download=True
        )

        self.valid = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=self.trans, download=True
        )

    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.valid
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)

        show_images(X.squeeze(1), nrows, ncols, titles=labels)
        
