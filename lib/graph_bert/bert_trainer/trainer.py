import abc
from collections import defaultdict

import attr
import dgl.dataloading
import torch.optim.optimizer

from torch import nn

from lib.graph_bert.bert_trainer.common import get_mask
from lib.graph_bert.bert_trainer.metric_collector import MLMetricCollector
from lib.graph_bert.nets.transform_block import GraphTransformBlockBase, GraphBertConfig
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import FEATURE_COLUMN


@attr.s
class TrainParamsConfig:
    mask_p: float = attr.ib()

    epoch: int = attr.ib()
    loss: nn.Module = attr.ib()


class GraphBertTrainerBase:
    def __init__(
        self,
        train_params: TrainParamsConfig,
    ):
        self.train_params = train_params

        self.metrics_collector = MLMetricCollector()

    @abc.abstractmethod
    def get_mask(self, batch_index: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def train_embeddings(self):
        pass

    @abc.abstractmethod
    def train_epoch(self):
        pass

    @abc.abstractmethod
    def train_label(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def test_model(self):
        pass

    @abc.abstractmethod
    def collect_train_metrics(self):
        pass


class GraphBertTrainer(GraphBertTrainerBase):
    def __init__(
        self,
        train_params: TrainParamsConfig,
        bert: GraphTransformBlockBase,
        bert_optimizer: torch.optim.optimizer,
        train_dataloader: dgl.dataloading.GraphCollator,
        test_dataloader: dgl.dataloading.GraphCollator,
    ):
        super().__init__(train_params)

        self.bert = bert
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_mask(self, batch_index: torch.Tensor) -> torch.Tensor:
        return get_mask(batch_index.shape, self.train_params.mask_p)

    def train_epoch(self):
        for g, label in self.train_dataloader:

            h_indexes_raw: torch.Tensor = g.ndata[FEATURE_COLUMN]
            e_indexes_raw: torch.Tensor = g.edata[FEATURE_COLUMN]

            mask_h = self.get_mask(h_indexes_raw)

            h_indexes = torch.clone(h_indexes_raw)
            h_indexes[mask_h] = 0

            embed_h, embed_e = self.model.forward(g, h_indexes, e_indexes_raw)

            embed_h[mask_h]
