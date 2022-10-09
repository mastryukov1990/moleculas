import abc
from abc import ABCMeta

import torch
from torch import nn

from lib.graph_bert.common import sum_tensors
from lib.logger import Logger

logger = Logger(__name__)


class AddLayerBase(metaclass=ABCMeta):
    @abc.abstractmethod
    @classmethod
    def aggregate(cls, x: torch.Tensor, x_add: torch.Tensor):
        pass

    @abc.abstractmethod
    @classmethod
    def forward(cls, x: torch.Tensor, x_add: torch.Tensor):
        pass


class AddLayerStable(AddLayerBase):
    @abc.abstractmethod
    def aggregate(self, x: torch.Tensor, x_add: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor, x_add: torch.Tensor) -> torch.Tensor:
        return self.aggregate(x, x_add)


class SumAddLayerStable(AddLayerStable):
    def aggregate(self, x: torch.Tensor, x_add: torch.Tensor) -> torch.Tensor:
        return sum_tensors(x, x_add)
