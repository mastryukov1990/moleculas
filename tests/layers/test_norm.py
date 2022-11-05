import torch
from hydra import compose, initialize

from lib.graph_bert.layers.layers.norm import BatchNorm

from lib.graph_bert.nets.config import cfg


def test_batch_norm():

    config = (
        cfg.transformer_block_config.graph_transformer_layer_config.e_branch_config.pre_layer_norm
    )
    x_batch = torch.ones([2, config.in_dim])

    norm = BatchNorm(config)
    target = norm.forward(x_batch)
    assert list(target.size()) == [2, config.in_dim]
