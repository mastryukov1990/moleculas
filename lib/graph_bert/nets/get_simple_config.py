from lib.graph_bert.layers.attention_blocks.base import MultiHeadAttentionLayerConfig
from lib.graph_bert.layers.blocks.branch import BranchFFNConfig
from lib.graph_bert.layers.blocks.fully_connected import FullyConnectedConfig
from lib.graph_bert.layers.blocks.graph_transformer import GraphTransformerLayerConfig
from lib.graph_bert.layers.config.config_base import ReadOutConfig
from lib.graph_bert.layers.layers.norm import NormConfig
from lib.graph_bert.layers.layers.o_layer import OutputAttentionLayerConfig
from lib.graph_bert.layers.mlp_readout_layer import MLPConfig
from lib.graph_bert.nets.transform_block import GraphBertTransformerConfig


def get_bert_config_simple(
    hidden_dim_attention=768,
    hidden_dim=768,
    fcc_hide=768,
    num_heads=12,
    out_dim=768,
    num_transforms=12,
    pos_enc_dim=37,
):
    multy_head_attention_conf = MultiHeadAttentionLayerConfig(
        in_dim=hidden_dim, num_heads=num_heads, out_dim=hidden_dim // num_heads
    )

    output_attention_config = OutputAttentionLayerConfig(
        in_dim=hidden_dim_attention,
        out_dim=hidden_dim,
    )

    pre_layer_norm = NormConfig(in_dim=hidden_dim)
    pre_batch_norm = NormConfig(in_dim=hidden_dim)

    fully_connected_config = FullyConnectedConfig(
        in_dim=hidden_dim, hidden_dim=fcc_hide, out_dim=hidden_dim
    )

    post_layer_norm = NormConfig(in_dim=hidden_dim)
    post_batch_norm = NormConfig(in_dim=hidden_dim)

    h_branch_config = BranchFFNConfig(
        fully_connected_config=fully_connected_config,
        output_attention_config=output_attention_config,
        pre_layer_norm=pre_layer_norm,
        pre_batch_norm=pre_batch_norm,
        post_layer_norm=post_layer_norm,
        post_batch_norm=post_batch_norm,
    )

    return GraphBertTransformerConfig(
        graph_transformer_layer_config=GraphTransformerLayerConfig(
            multy_head_attention_conf=multy_head_attention_conf,
            h_branch_config=h_branch_config,
            e_branch_config=h_branch_config,
        ),
        graph_transformer_layer_config_out=GraphTransformerLayerConfig(
            multy_head_attention_conf=multy_head_attention_conf,
            h_branch_config=h_branch_config,
            e_branch_config=h_branch_config,
        ),
        read_out_config=ReadOutConfig(),
        mlp_layer_config=MLPConfig(in_dim=hidden_dim, out_dim=out_dim),
        num_transforms=num_transforms,
        pos_enc_dim=pos_enc_dim,
    )
