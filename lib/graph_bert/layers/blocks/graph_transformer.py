import abc
from abc import ABCMeta

import dgl
import torch
import torch.nn as nn

from lib.graph_bert.layers.attention_blocks.base import (
    MultiHeadAttentionLayerBase,
    MultiHeadAttentionLayerConfig,
)
from lib.graph_bert.layers.blocks.branch import BranchFFNBase, BranchFFNConfig
from lib.graph_bert.layers.config.config_base import *
from lib.logger import Logger

logger = Logger(__name__)


class GraphTransformerLayerConfig(MultiHeadAttentionLayerConfig, BranchFFNConfig):
    pass


class GraphTransformerLayerBase(nn.Module, metaclass=ABCMeta):
    H_BRANCH_FFN = BranchFFNBase
    E_BRANCH_FFN = BranchFFNBase
    MULTY_HEAD_ATTENTION_LAYER = MultiHeadAttentionLayerBase

    @abc.abstractmethod
    def __init__(self, config: GraphTransformerLayerConfig):
        super().__init__()
        self.config = config


class GraphTransformerLayer(GraphTransformerLayerBase):
    H_BRANCH_FFN = BranchFFNBase
    E_BRANCH_FFN = BranchFFNBase
    MULTY_HEAD_ATTENTION_LAYER = MultiHeadAttentionLayerBase

    def __init__(self, config: GraphTransformerLayerConfig):
        super().__init__(config)

        self.attention: MultiHeadAttentionLayerBase = self.MULTY_HEAD_ATTENTION_LAYER(
            config
        )
        self.h_branch: BranchFFNBase = self.H_BRANCH_FFN(config)
        self.e_branch: BranchFFNBase = self.E_BRANCH_FFN(config)

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention.forward(g, h, e)
        logger.info(f"[{__name__}] h_attn_out = {h_attn_out.shape} ")

        h = self.h_branch.forward(h, h_in1)
        e = self.e_branch.forward(e, e_in1)

        return h, e

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )
