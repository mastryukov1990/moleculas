from abc import ABCMeta
from typing import Dict, Any, Callable

import dgl
import torch
from hydra.core.config_store import ConfigStore

from lib.graph_bert.layers.config.config_base import Readout, ReadOutConfig

LAST_NODE_FIELD = "h"

ReadoutConfigName = "readout"
ReadoutConfigGroup = "readout"


class ReadOutBase:
    READOUT_METHODS: Dict[Readout, Callable[[dgl.DGLHeteroGraph, str], Any]] = {
        Readout.sum: dgl.sum_nodes,
        Readout.max: dgl.max_nodes,
        Readout.mean: dgl.mean_nodes,
    }
    DEFAULT_METHOD: Callable[[dgl.DGLHeteroGraph, str], Any] = dgl.mean_nodes

    def __init__(self, config: ReadOutConfig):
        self.config = config

    def agg_graph(self, g: dgl.DGLHeteroGraph, h: torch.Tensor) -> torch.Tensor:
        pass


class ReadOut(ReadOutBase):
    def agg_graph(self, g: dgl.DGLHeteroGraph, h: torch.Tensor) -> torch.Tensor:
        g.ndata[LAST_NODE_FIELD] = h
        method = self.READOUT_METHODS.get(self.config.readout, self.DEFAULT_METHOD)
        return method(g, LAST_NODE_FIELD)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=ReadoutConfigGroup,
        name=ReadoutConfigName,
        node=ReadOutConfig,
    )
