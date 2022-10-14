import torch

from lib.graph_bert.layers.layers.linear_layer import LinearLayerConfig
from lib.graph_bert.layers.layers.o_layer import (
    OutputAttentionLayer,
    OutputAttentionLayerConfig,
)


def test_attention_layer():
    config = OutputAttentionLayerConfig(in_dim=10, out_dim=20)
    batch = 3

    x = torch.ones([batch, config.in_dim])

    layer = OutputAttentionLayer(config)
    target = layer.forward(x)
    assert list(target.shape) == [batch, config.out_dim]
