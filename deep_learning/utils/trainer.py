"""
Non concise way to implement Trainer like hugging face or pytorch lightning
"""
import torch
from tqdm import tqdm
from .module import ExtendedModule

class Trainer:
    def __init__(self, max_epochs: int, use_gpu=False, gradient_clip_val=0) -> None:
        """Initialises `Trainer` object for training any `scratch` models 

        Args:
            max_epochs (int): Total number of epochs to train model
            use_gpu (bool, optional): Training moves to gpu. Defaults to False.
            gradient_clip_val (int, optional): Clips gradients to given value. Defaults to 0.
        """
        self.max_epochs = max_epochs
        if use_gpu:
            assert torch.cuda.is_available(), 'GPU is not present'
        self.use_gpu = use_gpu
        self.gradient_clip_val = gradient_clip_val

    # TODO: data should be some general type class which enforces minimum functions required
    def _prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader else 0)

    def _prepare_model(self, model: ExtendedModule):
        model.trainer = self # FIXME: What was I thinking here? ;d
        model.board.xlim = [0, self.max_epochs]
        if self.use_gpu:
            model.to('cuda')
        self.model = model

    def fit(self, model: ExtendedModule, data):
        self._prepare_data(data)
        self._prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        # TODO: this is stupid but works for now
        description = f"Training {str(type(model)).split('.')[-1][:-2]}"
        for self.epoch in tqdm(range(self.max_epochs), desc=description):
            self.fit_epoch()

    def _prepare_batch(self, batch):
        if self.use_gpu:
            batch = [batch.to('cuda')]

        return batch

    def clip_gradient(self, grad_clip_val, model: ExtendedModule):
        params = [p for p in model.params() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2)) for p in params)
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit_epoch(self):
        self.model.train()

        for batch in self.train_dataloader:
            loss = self.model.training_step(self._prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradient(self.gradient_clip_val, self.model)
                self.optim.step()

            self.train_batch_idx += 1

        if not self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self._prepare_batch(batch))

            self.val_batch_idx += 1
    