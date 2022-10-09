import abc

import torch
from torch import nn

from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.add import SumAddLayerStable, AddLayerBase
from lib.graph_bert.layers.layers.linear_layer import (
    LinearWithLeakyReLU,
    LinearLayerBase,
    LinearLayerConfig,
)


@attr.s
class FullyConnectedConfig(
    HiddenDimConfig,
    NumHiddenConfig,
    LinearLayerConfig,
    Config,
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
    MAIN_BLOCK = LinearLayerBase
    ADD_LAYER = AddLayerBase

    def __init__(self, config: FullyConnectedConfig):
        super().__init__(config)
        self.head: LinearLayerBase = self.MAIN_BLOCK(
            LinearLayerConfig(
                in_dim=config.in_dim,
                out_dim=config.hidden_dim,
                bias=config.bias,
                activation=config.activation,
                dropout=config.dropout,
                layer_norm=config.layer_norm,
                batch_norm=config.batch_norm,
            )
        )
        self.body: nn.ModuleList[LinearLayerBase] = nn.ModuleList(
            [
                self.MAIN_BLOCK(
                    LinearLayerConfig(
                        in_dim=config.hidden_dim,
                        out_dim=config.hidden_dim,
                        bias=config.bias,
                        activation=config.activation,
                        dropout=config.dropout,
                        layer_norm=config.layer_norm,
                        batch_norm=config.batch_norm,
                    )
                )
                for _ in range(config.num_hidden)
            ]
        )
        self.tail: LinearLayerBase = self.MAIN_BLOCK(
            LinearLayerConfig(
                in_dim=config.hidden_dim,
                out_dim=config.out_dim,
                bias=config.bias,
                activation=config.activation,
                dropout=config.dropout,
                layer_norm=config.layer_norm,
                batch_norm=config.batch_norm,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head.forward(x)
        for part in self.body:
            x = part.forward(x)
        self.tail.forward(x)

        return x


class FullyConnectedLeakyLayer(FullyConnectedBlock):
    MAIN_BLOCK = LinearWithLeakyReLU
    ADD_LAYER = SumAddLayerStable
