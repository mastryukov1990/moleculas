import attr

from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedConfig, FullyConnectedSoftMax,
)


@attr.s
class MaskClassifierConfig(FullyConnectedConfig):
    pass


class MaskClassifier(FullyConnectedSoftMax):
    pass
