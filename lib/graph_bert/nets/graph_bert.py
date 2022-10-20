import abc

import attr
import dgl
import torch
from torch import nn

from lib.graph_bert.nets.mask_classifier import MaskClassifierConfig, MaskClassifier
from lib.graph_bert.nets.readout_mlp_net import ReadOutMlpConfig, ReadOutMlpBase
from lib.graph_bert.nets.transform_block import GraphBertConfig, GraphTransformBlockBase


@attr.s
class GraphBertNetsConfig:
    transformer_block_config: GraphBertConfig = attr.ib()
    classifier_mask_config: MaskClassifierConfig = attr.ib()
    readout_config: ReadOutMlpConfig = attr.ib()


@attr.s
class GraphBertConfig(GraphBertNetsConfig):
    is_classifier_mask_config: bool = attr.ib(default=True)
    is_readout_config: bool = attr.ib(default=False)


class GraphBertBase(nn.Module):
    def __init__(self, config: GraphBertConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        pass


class GraphBertDefault(GraphBertBase):
    GRAPH_TRANSFORMER_BLOCK: GraphTransformBlockBase
    CLASSIFIER_MASK: MaskClassifier
    READOUT_MLP: ReadOutMlpBase

    def __init__(
        self,
        config: GraphBertConfig,
    ):
        super().__init__(config)

        self.graph_transformer_block = self.GRAPH_TRANSFORMER_BLOCK(
            self.config.transformer_block_config
        )
        self.classifier_mask = (
            self.MASK_CLASSIFIER(self.config.classifier_mask_config)
            if self.config.is_classifier_mask_config
            else None
        )

        self.readout_mlp = (
            self.READOUT_MLP(self.config.readout_config)
            if self.config.is_readout_config
            else None
        )

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor, e: torch.Tensor):
        h, e = self.graph_transformer_block
