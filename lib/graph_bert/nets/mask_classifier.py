from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedConfig,
    FullyConnectedSoftMax,
)


MaskClassifierConfigGroup = "mask_classifier_group"
MaskClassifierConfigName = "mask_classifier_name"


@dataclass
class MaskClassifierConfig(FullyConnectedConfig):
    pass


class MaskClassifier(FullyConnectedSoftMax):
    pass


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=MaskClassifierConfigGroup,
        name=MaskClassifierConfigName,
        node=MaskClassifierConfig,
    )
