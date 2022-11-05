import torch


def get_mask(shape: torch.Size, p=0.15) -> torch.Tensor:
    return torch.FloatTensor(shape[0], shape[1]).uniform_() > p
