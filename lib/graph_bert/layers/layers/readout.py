from abc import ABCMeta
from typing import Dict, Any, Callable

import dgl
import torch
from torch import nn

from lib.graph_bert.layers.config.config_base import Readout, ReadOutConfig

LAST_NODE_FIELD = "h"


class ReadOutBase:

    READOUT_METHODS: Dict[
        Readout, Callable[[dgl.DGLHeteroGraph, str], Any]
    ] = {
        Readout.SUM: dgl.sum_nodes,
        Readout.MAX: dgl.max_nodes,
        Readout.MEAN: dgl.mean_nodes,
    }
    DEFAULT_METHOD: Callable[[dgl.DGLHeteroGraph, str], Any] = dgl.mean_nodes

    def __init__(self, config: ReadOutConfig):
        self.config = config

    def agg_graph(self, g: dgl.DGLHeteroGraph, h: torch.Tensor):
        pass


class ReadOut(ReadOutBase):
    def agg_graph(self, g: dgl.DGLHeteroGraph, h: torch.Tensor):
        g.ndata[LAST_NODE_FIELD] = h
        method = self.READOUT_METHODS.get(self.config.readout, self.DEFAULT_METHOD)

        return method(g, LAST_NODE_FIELD)
