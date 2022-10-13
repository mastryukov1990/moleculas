import abc
from abc import ABCMeta

from lib.graph_bert.layers.config.config_base import (
    HiddenDimConfig,
    InDimConfig,
    OutDimConfig,
)


class ComposeInBlock(InDimConfig, OutDimConfig):
    pass


class ComposeInBlockTopologyBase:
    def __init__(
        self,
        config: ComposeInBlock,
        config_in_dim: InDimConfig,
        config_hidden_dim: HiddenDimConfig,
        config_out_dim: OutDimConfig,
    ):
        self.config = config
        self.config_in_dim = config_in_dim
        self.config_hidden_dim = config_hidden_dim
        self.config_out_dim = config_out_dim

    def pre_config(self, config):
        pass

    def hidden_config(self, config):
        pass

    def post_config(self, config):
        pass
