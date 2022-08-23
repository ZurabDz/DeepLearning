import torch


class Trainer:
    def __init__(self, max_epochs, use_gpu=0, gradient_clip_val=0) -> None:
        self.max_epochs = max_epochs
        if use_gpu:
            assert torch.cuda.is_available(), 'GPU is not present'
        self.use_gpu = use_gpu
        self.gradient_clip_val = gradient_clip_val

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.use_gpu:
            model.to('cuda')
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [batch.to('cuda')]

        return batch

    def clip_gradient(self, grad_clip_val, model):
        params = [p for p in model.params() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2)) for p in params)
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit_epoch(self):
        self.model.train()

        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradient(self.gradient_clip_val, self.model)
                self.optim.step()

            self.train_batch_idx += 1

        if not self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))

            self.val_batch_idx += 1
    