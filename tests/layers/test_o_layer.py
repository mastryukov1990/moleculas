import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

from lib.graph_bert.layers.layers.o_layer import (
    OutputAttentionLayer,
    OutputAttentionLayerConfig,
    OutputAttentionLayerConfigName,
)
from lib.graph_bert.nets.config import cfg


def test_attention_layer():

    config = (
        cfg.transformer_block_config.graph_transformer_layer_config.e_branch_config.output_attention_config
    )
    batch = 3

    x = torch.ones([batch, config.in_dim])

    layer = OutputAttentionLayer(config)
    target = layer.forward(x)
    assert list(target.shape) == [batch, config.out_dim]
