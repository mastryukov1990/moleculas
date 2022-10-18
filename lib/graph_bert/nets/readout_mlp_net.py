import abc

import attr
import dgl
import torch
from torch import nn

from lib.graph_bert.layers.config.config_base import ReadOutConfig
from lib.graph_bert.layers.layers.readout import ReadOutBase, ReadOut
from lib.graph_bert.layers.mlp_readout_layer import MLPBase, MLPConfig, MLPDefault


@attr.s
class ReadOutMlpConfig:
    read_out_config: ReadOutConfig = attr.ib()
    mlp_layer_config: MLPConfig = attr.ib()


class ReadOutMlpBase(nn.Module):
    READOUT = ReadOutBase
    MLP = MLPBase

    def __init__(self, config: ReadOutMlpConfig):
        super().__init__()
        self.config = config


class ReadOutMlp(ReadOutMlpBase):
    def __init__(self, config: ReadOutMlpConfig):
        super().__init__(config)

        self.readout = self.READOUT(config.read_out_config)
        self.mlp = self.MLP(config.mlp_layer_config)

    def forward(self, g: dgl.DGLHeteroGraph, h: torch.Tensor) -> torch.Tensor:

        return self.mlp.forward(self.readout.agg_graph(g, h))


class ReadOutMlpDefault(ReadOutMlp):
    READOUT = ReadOut
    MLP = MLPDefault
