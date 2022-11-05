import abc
import copy
from dataclasses import dataclass
from IPython.display import clear_output

import dgl.dataloading
import torch.optim.optimizer


from lib.graph_bert.bert_trainer.common import get_mask
from lib.graph_bert.bert_trainer.metric_collector import MLMetricCollector
from lib.graph_bert.nets.graph_bert import GraphBertBase
from lib.logger import Logger
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import FEATURE_COLUMN

import matplotlib.pyplot as plt

logger = Logger(__name__)

@dataclass
class TrainParamsConfig:
    mask_p: float = 0.5
    epoch: int = 2
    is_classifier: bool = True
    is_readout: bool = True
    is_jupyter: bool = False


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


class GraphBertTrainerDefault(GraphBertTrainerBase):
    def __init__(
        self,
        train_params: TrainParamsConfig,
        bert: GraphBertBase,
        bert_optimizer: torch.optim.Optimizer,
        classifier_loss: torch.nn.CrossEntropyLoss,
        readout_loss: torch.nn.CrossEntropyLoss,
        train_dataloader: dgl.dataloading.GraphCollator,
        test_dataloader: dgl.dataloading.GraphCollator,
    ):
        super().__init__(train_params)

        self.bert = bert
        self.optimizer = bert_optimizer

        self.classifier_loss = classifier_loss
        self.readout_loss = readout_loss

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_mask(self, batch_index: torch.Tensor) -> torch.Tensor:
        return get_mask(batch_index.shape, self.train_params.mask_p)

    def train(self):
        for i in range(self.train_params.epoch):
            self.train_epoch()


    def train_epoch(self):

        for g, label in self.train_dataloader:
            self.train_batch(g, label)

        self.metrics_collector.update_train_collector()
        logger.info_metrics(self.metrics_collector.train_metric_collector.get_metrics())

        if self.train_params.is_jupyter:

            plt.figure()
            for i, v in self.metrics_collector.train_metric_collector.get_metrics().items():
                plt.plot(v, label=i)
            plt.legend()
            plt.grid()
            plt.show()
            clear_output(wait=True)



    def train_batch(self, g: dgl.DGLHeteroGraph, label: torch.Tensor):
        classifier_loss = 0
        readout_loss = 0

        h_indexes_raw: torch.Tensor = g.ndata[FEATURE_COLUMN]
        e_indexes_raw: torch.Tensor = g.edata[FEATURE_COLUMN]


        mask_h = self.get_mask(h_indexes_raw)

        classifier_h, readout_h = self.bert.forward(
            g, h_indexes_raw, e_indexes_raw, mask_h
        )

        if self.train_params.is_classifier:
            labels = g.ndata[FEATURE_COLUMN][mask_h]
            classifier_loss = self.classifier_loss(
                classifier_h, labels,
            ).mean()
            # torch.argmax(classifier_h, 1)

        if self.train_params.is_readout:
            readout_loss = self.readout_loss(readout_h, label)

        loss = classifier_loss + readout_loss
        loss.backward()
        self.optimizer.step()

        self.metrics_collector.train_metric_collector.collect_batch(
            "loss", copy.deepcopy(loss.item()), 1
        )
