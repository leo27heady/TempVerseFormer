import sys
from abc import ABC, abstractmethod

import torch


def seed_generator():
    """
    Fix seeds to allow for stochastic elements such as
    dropout to be reproduced exactly in activation
    recomputation in the backward pass.

    From RevViT.
    """

    # randomize seeds
    # use cuda generator if available
    if (
        hasattr(torch.cuda, "default_generators")
        and len(torch.cuda.default_generators) > 0
    ):
        # GPU
        device_idx = torch.cuda.current_device()
        seed = torch.cuda.default_generators[device_idx].seed()
    else:
        # CPU
        seed = int(torch.seed() % sys.maxsize)
    
    return 42 #seed


class ReversibleModule(torch.nn.Module, ABC):
    seeds = {}

    @abstractmethod
    def backward_pass(self, y, dy):
        pass
    
    def seed_cuda(self, key):
        self.seeds[key] = seed_generator()
        torch.manual_seed(self.seeds[key])


class NotReversibleModule(torch.nn.Module, ABC):
    seeds = {}

    @abstractmethod
    def forward_for_backward(self, x):
        pass
    
    def seed_cuda(self, key):
        if key in self.seeds:
            torch.manual_seed(self.seeds.pop(key))
        else:
            self.seeds[key] = seed_generator()
            torch.manual_seed(self.seeds[key])
