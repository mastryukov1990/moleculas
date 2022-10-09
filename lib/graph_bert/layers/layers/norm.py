import abc
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from lib.graph_bert.layers.config.config_base import InDimConfig
from lib.logger import Logger

logger = Logger(__name__)


class NormConfig(InDimConfig):
    pass


class NormBase(nn.Module, metaclass=ABCMeta):
    NORM = nn.Module

    @abstractmethod
    def __init__(self, config: NormConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def norm(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class LayerNormStable(NormBase):
    NORM = nn.LayerNorm

    def __init__(self, config: NormConfig):
        super().__init__(config)

        self.norm_layer = self.NORM(config.in_dim)

    def norm(self, x: torch.Tensor):
        return self.norm_layer(x)

    def forward(self, x: torch.Tensor):
        shape = x.shape

        logger.info(f"[{__name__}] {self.NORM.__class__.__name__}: x shape = {x.shape}")
        x = self.norm(x)

        assert shape == x.shape

        return x


class BatchNormStable(LayerNormStable):
    NORM = nn.BatchNorm1d
