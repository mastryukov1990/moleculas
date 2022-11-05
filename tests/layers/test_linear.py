import torch
from hydra import initialize, compose

from lib.graph_bert.layers.layers.common import LayersConfig
from lib.graph_bert.layers.layers.linear_layer import (
    LinearWithSoftMax,
    LinearLayerConfig,
    LinearWithLeakyReLU,
    LinearWithSigmoid,
)
from lib.graph_bert.nets.config import cfg


def test_linear():
    config = (
        cfg.transformer_block_config.graph_transformer_layer_config.e_branch_config.fully_connected_config
    )

    batch = 2

    x = torch.ones([batch, config.in_dim])
    for c in [LinearWithLeakyReLU, LinearWithSoftMax, LinearWithSigmoid]:
        k = c(config)
        target = k.forward(x)
        assert list(target.shape) == [batch, config.out_dim]
