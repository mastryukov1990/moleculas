import abc

import attr

from abc import ABCMeta

import torch
from torch import nn

from lib.graph_bert.layers.blocks.graph_transformer import (
    GraphTransformerLayerConfig,
    GraphTransformerLayer,
    GraphTransformerLayerBase,
)
from lib.graph_bert.layers.config.config_base import (
    PosEncDim,
    MaxWlRoleIndex,
    NumBondType,
    NumAtomType,
    InFeatDropout,
    NumTransformsConfig,
)
from lib.graph_bert.layers.mlp_readout_layer import (
    MLPReadoutConfig,
    MLPReadout,
    MLPReadoutBase,
)


@attr.s
class GraphNets:
    graph_transformer_layer_config: GraphTransformerLayerConfig
    graph_transformer_layer_config_out: GraphTransformerLayerConfig
    mlp_layer_config: MLPReadoutConfig

    @property
    def in_dim(self):
        return self.graph_transformer_layer_config.multy_head_attention_conf.in_dim


class GraphBertConfig(
    GraphNets,
    NumTransformsConfig,
    PosEncDim,
    MaxWlRoleIndex,
    NumAtomType,
    NumBondType,
    InFeatDropout,
):
    pass


class GraphBertBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: GraphBertConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class GraphBert(nn.Module):
    GRAPH_TRANSFORMER_LAYER = GraphTransformerLayerBase
    MLP_READOUT = MLPReadoutBase

    def __init__(self, config: GraphBertConfig):
        super().__init__()

        if self.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(config.pos_enc_dim, config.in_dim)

        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(
                config.max_wl_role_index, config.in_dim
            )

        self.embedding_h = nn.Embedding(config.num_atom_type, config.in_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(config.num_bond_type, config.in_dim)

        self.in_feat_dropout = nn.Dropout(config.in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                self.GRAPH_TRANSFORMER_LAYER(config.graph_transformer_layer_config)
                for _ in range(config.num_transforms - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(config.graph_transformer_layer_config_out)
        )
        self.MLP_layer = self.MLP_READOUT(config.mlp_layer_config)

    def forward(self, x: torch.Tensor):
        h = self.embedding_h(x)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

