import abc

import torch
from torch import nn

from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.add import SumAddLayerStable, AddLayerBase
from lib.graph_bert.layers.layers.linear_layer import (
    LinearWithLeakyReLU,
    LinearLayerBase,
)


@attr.s
class FullyConnectedConfig(
    InDimConfig, OutDimConfig, HiddenDimConfig, NumHiddenConfig, Config
):
    pass


class FullyConnectedBlockBase(nn.Module, metaclass=abc.ABCMeta):
    MAIN_BLOCK: nn.Module
    ADD_LAYER: AddLayerBase

    def __init__(self, config: FullyConnectedConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor):
        pass


class FullyConnectedBlock(FullyConnectedBlockBase):
    MAIN_BLOCK = LinearLayerBase()
    ADD_LAYER = AddLayerBase()

    def __init__(self, config: FullyConnectedConfig):
        super().__init__(config)
        self.head: LinearLayerBase = self.MAIN_BLOCK(config.in_dim, config.num_hidden)
        self.body: nn.ModuleList[LinearLayerBase] = nn.ModuleList(
            [
                self.MAIN_BLOCK(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_hidden)
            ]
        )
        self.tail: LinearLayerBase = self.MAIN_BLOCK(config.hidden_dim, config.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head.forward(x)
        for part in self.body:
            x = part.forward(x)
        self.tail.forward(x)

        return x


class FullyConnectedLeakyLayer(FullyConnectedBlock):
    MAIN_BLOCK = LinearWithLeakyReLU
    ADD_LAYER = SumAddLayerStable
