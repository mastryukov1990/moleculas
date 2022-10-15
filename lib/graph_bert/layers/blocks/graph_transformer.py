import abc
from abc import ABCMeta

import dgl
import torch
import torch.nn as nn

from lib.graph_bert.layers.attention_blocks.base import (
    MultiHeadAttentionLayerBase,
    MultiHeadAttentionLayerConfig,
    MultiHeadAttentionLayer,
)
from lib.graph_bert.layers.attention_blocks.multy_head_attention import (
    MultiHeadAttentionLayerDefault,
)
from lib.graph_bert.layers.blocks.branch import (
    BranchFFNBase,
    BranchFFNConfig,
    BranchFFNAttentionDefault,
)
from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.o_layer import OutputAttentionLayerConfig
from lib.logger import Logger
import attr

logger = Logger(__name__)


@attr.s
class GraphTransformerLayerConfig:
    multy_head_attention_conf: MultiHeadAttentionLayerConfig = attr.ib()
    h_branch_config: BranchFFNConfig = attr.ib()
    e_branch_config: BranchFFNConfig = attr.ib()


class GraphTransformerLayerBase(nn.Module, metaclass=ABCMeta):
    H_BRANCH_FFN = BranchFFNBase
    E_BRANCH_FFN = BranchFFNBase
    MULTY_HEAD_ATTENTION_LAYER = MultiHeadAttentionLayerBase

    @abc.abstractmethod
    def __init__(self, config: GraphTransformerLayerConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        pass


class GraphTransformerLayer(GraphTransformerLayerBase):
    def __init__(self, config: GraphTransformerLayerConfig):
        super().__init__(config)

        self.attention: MultiHeadAttentionLayerBase = self.MULTY_HEAD_ATTENTION_LAYER(
            config.multy_head_attention_conf
        )
        self.h_branch: BranchFFNBase = self.H_BRANCH_FFN(config.h_branch_config)
        self.e_branch: BranchFFNBase = self.E_BRANCH_FFN(config.e_branch_config)

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention.forward(g, h, e)

        h = self.h_branch.forward(h_attn_out, h_in1)
        e = self.e_branch.forward(e_attn_out, e_in1)

        return h, e

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )


class GraphTransformerLayerDefault(GraphTransformerLayer):
    H_BRANCH_FFN = BranchFFNAttentionDefault
    E_BRANCH_FFN = BranchFFNAttentionDefault
    MULTY_HEAD_ATTENTION_LAYER = MultiHeadAttentionLayerDefault
