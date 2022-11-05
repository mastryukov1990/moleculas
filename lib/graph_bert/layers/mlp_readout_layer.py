"""
    MLP Layer used after graph vector representation
"""
from typing import Dict

import attr
import torch
from torch import nn

from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedBlock,
    FullyConnectedConfig,
    FullyConnectedBlockBase,
    FullyConnectedLeakyLayer,
)


class MLPConfig(FullyConnectedConfig):
    hidden_dim: int = 256

    num_hidden: int = 10

    in_dim: int = 12
    out_dim: int = 128
    bias: bool = True
    activation: bool = True
    dropout: bool = True
    layer_norm: bool = False
    batch_norm: bool = True


class MLPBase(nn.Module):
    FC_LAYER = FullyConnectedBlockBase

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.net = self.FC_LAYER(config)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MLPDefault(MLPBase):
    FC_LAYER = FullyConnectedLeakyLayer
