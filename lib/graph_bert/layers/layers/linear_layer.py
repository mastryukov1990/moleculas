from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from lib.graph_bert.layers.config.block_configs import ComposeInBlock
from lib.graph_bert.layers.config.config_base import *

LinearLayerConfigName = "linear_layer"


@attr.s
class LinearLayerConfig(
    ComposeInBlock,
    BiasConfig,
    ActivationConfig,
    DropoutConfig,
    LayerNormConfig,
    BatchNormConfig,
    CopyConfig,
):
    pass


class LinearLayerBase(nn.Module):
    NEURON_LAYER: nn.Module
    ACTIVATION: nn.Module
    DROPOUT: nn.Module
    LAYER_NORM: nn.Module
    BATCH_NORM: nn.Module

    @abstractmethod
    def __init__(self, config: LinearLayerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class LinearLayerInit(LinearLayerBase):
    NEURON_LAYER = nn.Linear
    ACTIVATION = nn.ReLU
    DROPOUT = nn.Dropout
    LAYER_NORM = nn.LayerNorm
    BATCH_NORM = nn.BatchNorm1d

    def __init__(self, config: LinearLayerConfig):
        super().__init__(config=config)
        self.neuron_layer = self.NEURON_LAYER(
            config.in_dim, config.out_dim, bias=config.bias
        )
        self.activation = self.ACTIVATION() if config.activation else None
        self.dropout = self.DROPOUT(config.dropout) if config.dropout else None
        self.layer_norm = self.LAYER_NORM(config.out_dim) if config.layer_norm else None
        self.batch_norm = self.BATCH_NORM(config.out_dim) if config.batch_norm else None

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class LinearActivationNormalization(LinearLayerInit):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.neuron_layer.forward(x)

        if self.activation:
            x = self.activation.forward(x)

        if self.dropout:
            x = self.dropout.forward(x)

        if self.layer_norm:
            return self.layer_norm.forward(x)

        if self.batch_norm:
            return self.batch_norm.forward(x)

        return x


class LinearWithSigmoid(LinearActivationNormalization):
    ACTIVATION = nn.Sigmoid


class LinearWithLeakyReLU(LinearActivationNormalization):
    ACTIVATION = nn.LeakyReLU


class LinearWithSoftMax(LinearActivationNormalization):
    ACTIVATION = nn.Softmax
