import random

import numpy as np
import torch


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_gradients(self):
    grads = []
    for param in self.parameters():
        if param.grad is not None:
            grad = param.grad
        else:
            grad = torch.zeros(param.shape, dtype=param.dtype, device=param.device)
        grads.append(grad.view(-1))
    grads = torch.cat(grads).clone()
    return grads

def reduce_grad(grad):
    grad = grad.norm(2, dim=-1)
    if grad.dim() > 0:
        grad = grad.sum(dim=0)
    return grad
