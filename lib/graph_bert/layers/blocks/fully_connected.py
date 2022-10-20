import abc

import torch
from torch import nn

from lib.graph_bert.layers.config.block_configs import ComposeInBlockTopologyBase
from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.linear_layer import (
    LinearWithLeakyReLU,
    LinearLayerBase,
    LinearLayerConfig,
    LinearWithSoftMax,
)


@attr.s
class FullyConnectedConfig(
    HiddenDimConfig,
    NumHiddenConfig,
    LinearLayerConfig,
    Config,
):
    pass


class ComposeInBlockTopologyBaseFullyConnected(ComposeInBlockTopologyBase):
    def pre_config(self, config: FullyConnectedConfig) -> LinearLayerConfig:
        config = config.get_copy()
        config.in_dim = self.config_in_dim.in_dim
        config.out_dim = self.config_hidden_dim.hidden_dim
        return config

    def hidden_config(self, config: FullyConnectedConfig):
        config = config.get_copy()
        config.in_dim = self.config_hidden_dim.hidden_dim
        config.out_dim = self.config_hidden_dim.hidden_dim
        return config

    def post_config(self, config: FullyConnectedConfig):
        config = config.get_copy()
        config.in_dim = self.config_hidden_dim.hidden_dim
        config.out_dim = self.config_out_dim.out_dim
        return config


class FullyConnectedBlockBase(nn.Module, metaclass=abc.ABCMeta):
    MAIN_BLOCK: LinearLayerBase
    TAIL_BLOCK: LinearLayerBase
    COMPOSE_BLOCK_TOPOLOGY = ComposeInBlockTopologyBase

    def __init__(self, config: FullyConnectedConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor):
        pass


class FullyConnectedBlock(FullyConnectedBlockBase):
    def __init__(self, config: FullyConnectedConfig):
        super().__init__(config)
        compose_block_config = self.COMPOSE_BLOCK_TOPOLOGY(
            config,
            config_in_dim=InDimConfig(config.in_dim),
            config_out_dim=OutDimConfig(config.out_dim),
            config_hidden_dim=HiddenDimConfig(config.hidden_dim),
        )

        self.head: LinearLayerBase = self.MAIN_BLOCK(
            compose_block_config.pre_config(self.config),
        )
        self.body: nn.ModuleList[LinearLayerBase] = nn.ModuleList(
            [
                self.MAIN_BLOCK(compose_block_config.hidden_config(self.config))
                for _ in range(config.num_hidden)
            ]
        )
        self.tail: LinearLayerBase = self.TAIL_BLOCK(
            compose_block_config.post_config(self.config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head.forward(x)
        for part in self.body:
            x = part.forward(x)
        x = self.tail.forward(x)

        return x


class FullyConnectedLeakyLayer(FullyConnectedBlock):
    MAIN_BLOCK = LinearWithLeakyReLU
    TAIL_BLOCK = LinearWithLeakyReLU
    COMPOSE_BLOCK_TOPOLOGY = ComposeInBlockTopologyBaseFullyConnected


class FullyConnectedSoftMax(FullyConnectedBlock):
    MAIN_BLOCK = LinearWithLeakyReLU
    TAIL_BLOCK = LinearWithSoftMax
    COMPOSE_BLOCK_TOPOLOGY = ComposeInBlockTopologyBaseFullyConnected
