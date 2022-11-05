import abc

import attr

from abc import ABCMeta
from dataclasses import dataclass
import dgl
import torch
from hydra.core.config_store import ConfigStore
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
    MLPDefault,
    MLPBase,
)

GraphBertTransformerConfigGroup = "transform_config_group"
GraphBertTransformerConfigName = "transform_config_name"


@dataclass
class GraphNetsConfig:
    graph_transformer_layer_config: GraphTransformerLayerConfig = (
        GraphTransformerLayerConfig()
    )
    graph_transformer_layer_config_out: GraphTransformerLayerConfig = (
        GraphTransformerLayerConfig()
    )


@dataclass
class GraphBertTransformerConfig(
    NumTransformsConfig,
    PosEncDim,
    MaxWlRoleIndex,
    NumAtomType,
    NumBondType,
    InFeatDropout,
    GraphNetsConfig,
):
    pass


class GraphTransformBlockBase(nn.Module):
    def __init__(self, config: GraphBertTransformerConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        pass


class GraphTransformBlock(GraphTransformBlockBase):
    GRAPH_TRANSFORMER_LAYER = GraphTransformerLayerBase

    def __init__(self, config: GraphBertTransformerConfig):
        super().__init__(config=config)

        self.embedding_h = nn.Embedding(
            config.num_atom_type,
            config.graph_transformer_layer_config.multy_head_attention_conf.in_dim,
        )

        self.embedding_e = nn.Embedding(
            config.num_bond_type,
            config.graph_transformer_layer_config.multy_head_attention_conf.in_dim,
        )

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

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        h = h.squeeze(1)
        e = e.squeeze(1)

        # convnets
        for i, conv in enumerate(self.layers):

            h, e = conv.forward(g, h, e)

        return h, e


class GraphTransformBlockDefault(GraphTransformBlock):
    GRAPH_TRANSFORMER_LAYER = GraphTransformerLayerDefault


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=GraphBertTransformerConfigGroup,
        name=GraphBertTransformerConfigName,
        node=GraphBertTransformerConfig,
    )
