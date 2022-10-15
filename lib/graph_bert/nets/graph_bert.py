import abc

import attr

from abc import ABCMeta

import dgl
import torch
from torch import nn

from lib.graph_bert.layers.blocks.graph_transformer import (
    GraphTransformerLayerConfig,
    GraphTransformerLayer,
    GraphTransformerLayerBase,
    GraphTransformerLayerDefault,
)
from lib.graph_bert.layers.config.config_base import (
    PosEncDim,
    MaxWlRoleIndex,
    NumBondType,
    NumAtomType,
    InFeatDropout,
    NumTransformsConfig,
    ReadOutConfig,
    FromDictConfig,
)
from lib.graph_bert.layers.layers.readout import ReadOutBase, ReadOut
from lib.graph_bert.layers.mlp_readout_layer import (
    MLPConfig,
    MLP,
    MLPBase,
)


@attr.s
class GraphNetsConfig:
    graph_transformer_layer_config: GraphTransformerLayerConfig = attr.ib()
    graph_transformer_layer_config_out: GraphTransformerLayerConfig = attr.ib()
    read_out_config: ReadOutConfig = attr.ib()
    mlp_layer_config: MLPConfig = attr.ib()

    @property
    def in_dim(self):
        return self.graph_transformer_layer_config.multy_head_attention_conf.in_dim


@attr.s
class GraphBertConfig(
    GraphNetsConfig,
    NumTransformsConfig,
    PosEncDim,
    MaxWlRoleIndex,
    NumAtomType,
    NumBondType,
    InFeatDropout,
    FromDictConfig,
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
    READOUT = ReadOutBase
    MLP = MLPBase

    def __init__(self, config: GraphBertConfig):
        super().__init__()

        self.embedding_h = nn.Embedding(config.num_atom_type, config.in_dim)

        self.embedding_e = nn.Embedding(config.num_bond_type, config.in_dim)

        self.in_feat_dropout = nn.Dropout(config.in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                self.GRAPH_TRANSFORMER_LAYER(config.graph_transformer_layer_config)
                for _ in range(config.num_transforms - 1)
            ]
        )
        self.layers.append(
            self.GRAPH_TRANSFORMER_LAYER(config.graph_transformer_layer_config_out)
        )

        self.readout = self.READOUT(config.read_out_config)

        self.mlp = self.MLP(config.mlp_layer_config)

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        e = self.embedding_e(e)

        # convnets
        for i, conv in enumerate(self.layers):

            h, e = conv.forward(g, h, e)

        hg = self.readout.agg_graph(g, h)

        return self.mlp.forward(hg)


class GraphBertDefault(GraphBert):
    GRAPH_TRANSFORMER_LAYER = GraphTransformerLayerDefault
    READOUT = ReadOut
    MLP = MLP
