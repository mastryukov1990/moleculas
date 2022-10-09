import torch


def sum_tensors(x: torch.Tensor, x_add: torch.Tensor):
    assert x.shape == x_add.shape
    return x + x_add
