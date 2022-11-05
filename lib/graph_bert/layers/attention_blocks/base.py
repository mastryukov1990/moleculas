import abc
from abc import ABCMeta, abstractmethod
from typing import Tuple

import dgl
import torch
from torch import nn

from lib.graph_bert.layers.config.block_configs import ComposeInBlock
from lib.graph_bert.layers.config.config_base import *
from lib.logger import Logger

logger = Logger(__name__)

Q_H = "Q_h"
K_H = "K_h"
V_H = "V_h"
PROJ_E = "proj_e"
SCORE = "score"


@dataclass
class MultiHeadAttentionLayerConfig(ComposeInBlock, NumHeadsConfig, BiasConfig, Config):
    SECTIONS = [BASE_SECTION, MULTI_HEAD_ATTENTION_LAYER_SECTION]


class MultiHeadAttentionLayerBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: MultiHeadAttentionLayerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def propagate_attention(self, g: dgl.DGLHeteroGraph):
        pass

    @abstractmethod
    def reduce_attention(
        self, g: dgl.DGLHeteroGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class MultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    def __init__(self, config: MultiHeadAttentionLayerConfig):
        super().__init__(config)

        in_dim = config.in_dim

        bias = config.bias

        self.num_heads = config.num_heads
        self.out_dim = config.out_dim
        hidden_attention_dim = self.out_dim * self.num_heads

        self.Q = nn.Linear(in_dim, hidden_attention_dim, bias=bias)
        self.K = nn.Linear(in_dim, hidden_attention_dim, bias=bias)
        self.V = nn.Linear(in_dim, hidden_attention_dim, bias=bias)
        self.proj_e = nn.Linear(in_dim, hidden_attention_dim, bias=bias)

    def propagate_attention(self, g: dgl.DGLHeteroGraph):
        pass

    def reduce_attention(self, g: dgl.DGLHeteroGraph):
        pass

    def forward(
        self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_h = self.Q(h)
        k_h = self.K(h)
        v_h = self.V(h)

        logger.info(
            f"[{__name__}] Q_H shape = {q_h.shape},K_h shape = {k_h.shape},V_h shape = {v_h.shape}"
        )
        proj_e = self.proj_e(e)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata[Q_H] = q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata[K_H] = k_h.view(-1, self.num_heads, self.out_dim)
        g.ndata[V_H] = v_h.view(-1, self.num_heads, self.out_dim)
        g.edata[PROJ_E] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        return self.reduce_attention(g)
