from abc import abstractmethod

import torch
from hydra.core.config_store import ConfigStore

from lib.graph_bert.layers.layers.linear_layer import (
    LinearLayerInit,
    LinearLayerConfig,
)

ReadoutConfigGroup = "output_attention"
OutputAttentionLayerConfigName = "output_attention"


class OutputAttentionLayerConfig(LinearLayerConfig):
    pass


class OutputAttentionLayerBase(LinearLayerInit):
    def __init__(self, config: OutputAttentionLayerConfig):
        super().__init__(config)

    @abstractmethod
    def concat(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class OutputAttentionLayer(OutputAttentionLayerBase):
    def concat(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.config.in_dim)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.concat(x)

        if self.dropout:
            x = self.dropout(x)

        return self.neuron_layer(x)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=ReadoutConfigGroup,
        name=OutputAttentionLayerConfigName,
        node=OutputAttentionLayerConfig,
    )
