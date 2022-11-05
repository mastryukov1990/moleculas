import abc
from abc import ABCMeta

import torch
from hydra.core.config_store import ConfigStore
from torch import nn

from lib.graph_bert.common import sum_tensors
from lib.logger import Logger

logger = Logger(__name__)


class AddLayerBase(abc.ABC):
    @classmethod
    def aggregate(cls, x: torch.Tensor, x_add: torch.Tensor):
        pass

    @classmethod
    def forward(cls, x: torch.Tensor, x_add: torch.Tensor):
        pass


class SumAddLayer(AddLayerBase):
    @classmethod
    def forward(cls, x: torch.Tensor, x_add: torch.Tensor) -> torch.Tensor:
        return cls.aggregate(x, x_add)

    @classmethod
    def aggregate(cls, x: torch.Tensor, x_add: torch.Tensor) -> torch.Tensor:
        return sum_tensors(x, x_add)
