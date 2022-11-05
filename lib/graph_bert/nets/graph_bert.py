import abc
from dataclasses import dataclass
from typing import Tuple

import attr
import dgl
import torch
from torch import nn
from copy import copy

from lib.graph_bert.bert_trainer.common import get_mask
from lib.graph_bert.nets.mask_classifier import MaskClassifierConfig, MaskClassifier
from lib.graph_bert.nets.readout_mlp_net import (
    ReadOutMlpConfig,
    ReadOutMlpBase,
    ReadOutMlpDefault,
)
from lib.graph_bert.nets.transform_block import (
    GraphBertTransformerConfig,
    GraphTransformBlockDefault,
)


@dataclass
class GraphBertNetsConfig:
    transformer_block_config: GraphBertTransformerConfig = GraphBertTransformerConfig()


@dataclass
class GraphBertConfig:
    is_classifier_mask_config: bool = True
    is_readout_config: bool = False
    classifier_mask_config: MaskClassifierConfig = MaskClassifierConfig()
    readout_config: ReadOutMlpConfig = ReadOutMlpConfig()
    transformer_block_config: GraphBertTransformerConfig = GraphBertTransformerConfig()


class GraphBertBase(nn.Module):
    def __init__(self, config: GraphBertConfig):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class GraphBertDefault(GraphBertBase):
    GRAPH_TRANSFORMER_BLOCK = GraphTransformBlockDefault
    CLASSIFIER_MASK = MaskClassifier
    READOUT_MLP = ReadOutMlpDefault

    def __init__(
        self,
        config: GraphBertConfig,
    ):
        super().__init__(config)

        self.graph_transformer_block = self.GRAPH_TRANSFORMER_BLOCK(
            self.config.transformer_block_config
        )
        self.classifier_mask = (
            self.CLASSIFIER_MASK(self.config.classifier_mask_config)
            if self.config.is_classifier_mask_config
            else None
        )

        self.readout_mlp = (
            self.READOUT_MLP(self.config.readout_config)
            if self.config.is_readout_config
            else None
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        masked_h = torch.clone(h)
        masked_h[torch.logical_not(mask)] = 0

        t_h, t_e = self.graph_transformer_block.forward(g, masked_h, e)
        classifier_pred = (
            self.classifier_mask.forward(t_h[mask.squeeze(1)])
            if self.classifier_mask
            else torch.Tensor()
        )
        readout_pred = (
            self.readout_mlp.forward(g, t_h) if self.readout_mlp else torch.Tensor()
        )

        return classifier_pred, readout_pred
