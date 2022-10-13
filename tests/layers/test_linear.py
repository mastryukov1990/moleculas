import torch

from lib.graph_bert.layers.layers.linear_layer import (
    LinearWithSoftMax,
    LinearLayerConfig,
    LinearWithLeakyReLU,
    LinearWithSigmoid,
)


def test_linear():
    config = LinearLayerConfig()
    config.in_dim = 10
    config.out_dim = 20
    batch = 2

    x = torch.ones([batch, config.in_dim])
    for c in [LinearWithLeakyReLU, LinearWithSoftMax, LinearWithSigmoid]:
        k = c(config)
        target = k.forward(x)
        assert list(target.shape) == [batch, config.out_dim]
