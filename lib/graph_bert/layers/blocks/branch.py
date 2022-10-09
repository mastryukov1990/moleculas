from abc import ABCMeta

import attr
import torch
from torch import nn

from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedBlock,
    FullyConnectedBlockBase,
    FullyConnectedConfig,
)
from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.add import (
    AddLayerBase,
)
from lib.graph_bert.layers.layers.linear_layer import LinearLayerConfig
from lib.graph_bert.layers.layers.norm import NormBase, NormConfig
from lib.graph_bert.layers.layers.o_layer import (
    OutputAttentionLayerBase,
    OutputAttentionLayer,
)
from lib.logger import Logger

logger = Logger(__name__)


class BranchFFNConfig(
    FullyConnectedConfig,
    PreAddLayerConfig,
    PostAddLayerConfig,
    NormConfig,
):
    SECTIONS = [BASE_SECTION, BRANCH_FFN_SECTION]


class BranchFFNBase(nn.Module, metaclass=ABCMeta):
    ADD_LAYER = AddLayerBase
    OUTPUT_ATTENTION_LAYER = OutputAttentionLayerBase
    LAYER_NORM = NormBase
    BATCH_NORM = NormBase
    FULLY_CONNECTED_BLOCK = FullyConnectedBlockBase

    def __init__(self, config: BranchFFNConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, x_res: torch.Tensor):
        pass


class BranchFFN(BranchFFNBase):
    ADD_LAYER = AddLayerBase
    OUTPUT_ATTENTION_LAYER = OutputAttentionLayer
    LAYER_NORM = NormBase
    BATCH_NORM = NormBase
    FULLY_CONNECTED_BLOCK = FullyConnectedBlockBase

    def __init__(self, config: BranchFFNConfig):
        super().__init__(config)
        out_dim = config.out_dim

        self.output_attention_layer: OutputAttentionLayerBase = (
            self.OUTPUT_ATTENTION_LAYER(
                LinearLayerConfig(
                    out_dim=out_dim,
                    in_dim=out_dim,
                    bias=config.bias,
                    activation=config.activation,
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                    batch_norm=config.batch_norm,
                )
            )
        )

        self.pre_add_layer = self.ADD_LAYER if config.pre_add_layer else None
        self.post_add_layer = self.ADD_LAYER if config.post_add_layer else None

        self.pre_layer_norm: NormBase = (
            self.LAYER_NORM(config) if config.layer_norm else None
        )
        self.pre_batch_norm: NormBase = (
            self.BATCH_NORM(config) if config.batch_norm else None
        )

        self.fully_connected: FullyConnectedBlockBase = self.FULLY_CONNECTED_BLOCK(
            config
        )

        self.post_layer_norm: NormBase = (
            self.LAYER_NORM(config) if config.layer_norm else None
        )

        self.post_batch_norm: NormBase = (
            self.BATCH_NORM(config) if config.batch_norm else None
        )

    def forward(self, x: torch.Tensor, x_res: torch.Tensor):
        logger.info(f"[{__name__}] h_attn_out = {x.shape} ")
        x = self.output_attention_layer.forward(x)

        if self.pre_add_layer:
            x = self.pre_add_layer.forward(x, x_res)

        if self.pre_layer_norm:
            x = self.layer_norm.forward(x)

        if self.pre_batch_norm:
            x = self.batch_norm.forward(x)

        x_post = torch.Tensor()

        if self.post_add_layer:
            x_post = x

        if self.fully_connected:
            x = self.fully_connected.forward(x)

        if self.post_add_layer:
            x = self.post_add_layer.forward(x, x_post)

        if self.post_layer_norm:
            x = self.layer_norm.forward(x)

        if self.post_batch_norm:
            x = self.batch_norm.forward(x)

        return x
