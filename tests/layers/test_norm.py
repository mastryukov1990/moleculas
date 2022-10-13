import torch

from lib.graph_bert.layers.layers.norm import BatchNorm, NormConfig


def test_batch_norm():
    config = NormConfig()
    x_batch = torch.ones([2, config.in_dim])

    norm = BatchNorm(config)
    target = norm.forward(x_batch)
    assert list(target.size()) == [2, config.in_dim]
