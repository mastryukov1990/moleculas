import torch

from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedLeakyLayer,
    FullyConnectedConfig,
)


def test_fully_connected():
    config = FullyConnectedConfig()
    batch = 2

    x_branch = torch.ones([batch, config.in_dim])

    fcc = FullyConnectedLeakyLayer(config)
    target = fcc.forward(x_branch)
    assert list(target.shape) == [batch, config.out_dim]
