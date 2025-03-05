import torch
from torch.utils.data import IterableDataset

from ..config import DataConfig


class MockShapeDataset(IterableDataset):
    def __init__(self, device, config: DataConfig, latent: bool = False):
        self.device = device
        self.config = config
        self.latent = latent

    def __iter__(self):
        while True:
            current_batch_time = self.config.time_to_pred.max

            full_size = self.config.context_size + current_batch_time
            if self.latent:
                output_shape = (256, 1, 1)
            else:
                output_shape = (self.config.im_channels, self.config.render_window_size, self.config.render_window_size)

            yield (
                torch.arange(start=1, end=current_batch_time + 1, device=self.device), 
                torch.randn(self.config.batch_size, full_size, *output_shape, device=self.device).requires_grad_(),  # images
                torch.randn(self.config.batch_size, full_size, device=self.device),  # angles
                torch.randn(self.config.batch_size, full_size, 4, device=self.device)  # patterns
            )
