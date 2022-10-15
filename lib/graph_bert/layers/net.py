from abc import ABCMeta

import attr
from torch import nn

from lib.graph_bert.layers.blocks.graph_transformer import (
    GraphTransformerLayerBase,
    GraphTransformerLayerConfig,
)
from lib.graph_bert.layers.config.net_config import NetConfig
from lib.graph_bert.layers.mlp_readout_layer import MLP


class GraphBertBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config


class GraphBert(GraphBertBase):
    GRAPH_TRANSFORMER_LAYER = GraphTransformerLayerBase
    MLP_READOUT = MLP

    def __init__(self, config: NetConfig):
        super().__init__(config)

        if self.pos_enc_dim:
            self.embedding_lap_pos_enc = nn.Linear(
                config.pos_enc_dim, config.hidden_dim
            )

        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(
                config.max_wl_role_index, config.hidden_dim
            )

        self.embedding_h = nn.Embedding(
            config.in_dim, config.hidden_dim
        )  # node feat is an integer

        self.in_feat_dropout = nn.Dropout(config.in_feat_dropout)

        self.layers = nn.ModuleList(
            [self.GRAPH_TRANSFORMER_LAYER() for _ in range(self.num_transforms - 1)]
        )
        self.layers.append(self.GRAPH_TRANSFORMER_LAYER(config))
        self.MLP_layer = MLP(config)
