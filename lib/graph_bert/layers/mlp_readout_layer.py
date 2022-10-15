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
    pass


class MLPBase(nn.Module):
    FC_LAYER = FullyConnectedBlockBase

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.net = self.FC_LAYER(config)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MLP(MLPBase):
    FC_LAYER = FullyConnectedLeakyLayer
