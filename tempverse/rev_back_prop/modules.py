import sys
from abc import ABC, abstractmethod

import torch


def seed_generator() -> int:
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
    
    return seed
    
def employ_seed(custom_backward: bool, seeds: dict[str, int], key: int, safe: bool) -> None:
    # To be able to restore the same activations during backward as in forward
    if custom_backward and key in seeds:
        seed = seeds[key] if safe else seeds.pop(key)
        torch.manual_seed(seed)
    else:
        seeds[key] = seed_generator()
        torch.manual_seed(seeds[key])


class ReversibleModule(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        self.seeds = {}
        self.custom_backward = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def backward_pass(self, y, dy):
        pass
    
    def seed_cuda(self, key: str, safe: bool = False):
        employ_seed(self.custom_backward, self.seeds, key, safe)


class NotReversibleModule(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        self.seeds = {}
        self.custom_backward = True
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward_for_backward(self, x):
        pass
    
    def seed_cuda(self, key: str, safe: bool = False):
        employ_seed(self.custom_backward, self.seeds, key, safe)
