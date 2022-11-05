import torch

from lib.graph_bert.layers.blocks.branch import (
    BranchFFNAttentionDefault,
    BranchFFNConfig,
)
from lib.graph_bert.layers.blocks.fully_connected import FullyConnectedConfig
from lib.graph_bert.layers.layers.norm import NormConfig
from lib.graph_bert.layers.layers.o_layer import OutputAttentionLayerConfig


def test_branch():
    fcc_hide = 1024
    hidden_dim = 256
    batch = 2

    output_attention_config = OutputAttentionLayerConfig(out_dim=hidden_dim)

    pre_layer_norm = NormConfig(in_dim=hidden_dim)
    pre_batch_norm = NormConfig(in_dim=hidden_dim)

    fully_connected_config = FullyConnectedConfig(
        in_dim=hidden_dim, hidden_dim=fcc_hide, out_dim=hidden_dim
    )

    post_layer_norm = NormConfig(in_dim=hidden_dim)
    post_batch_norm = NormConfig(in_dim=hidden_dim)

    config = BranchFFNConfig(
        fully_connected_config=fully_connected_config,
        output_attention_config=output_attention_config,
        pre_layer_norm=pre_layer_norm,
        pre_batch_norm=pre_batch_norm,
        post_layer_norm=post_layer_norm,
        post_batch_norm=post_batch_norm,
    )

    x_batch = torch.ones([batch, output_attention_config.in_dim])
    x_res = torch.ones([batch, output_attention_config.out_dim])

    branch = BranchFFNAttentionDefault(config)
    target = branch.forward(x_batch, x_res)

    assert list(target.shape) == [batch, post_batch_norm.in_dim]
