import torch
from dgl.dataloading import GraphDataLoader
from torch import optim, nn

from lib.graph_bert.bert_trainer.trainer import (
    GraphBertTrainerDefault,
    TrainParamsConfig,
)
from lib.graph_bert.nets.config import cfg
from lib.graph_bert.nets.graph_bert import GraphBertDefault
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    MoleculeGraphBuilder,
)


def test_GraphBertTrainerDefault(molecules):
    batch = 10

    molecule_graph_builder = [
        MoleculeGraphBuilder.from_smile(
            molecule, atom_glossary, bond_glossary
        ).get_graph()
        for molecule in molecules
    ]
    labels = torch.ones(len(molecules)).type(torch.long)
    dataset = MoleculesDataset(molecule_graph_builder, labels)
    loader = GraphDataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
    )

    bert = GraphBertDefault(config=cfg)
    optimizer = optim.Adam(bert.parameters())

    trainer = GraphBertTrainerDefault(
        train_params=TrainParamsConfig(epoch=10),
        bert=bert,
        bert_optimizer=optimizer,
        classifier_loss=nn.CrossEntropyLoss(),
        readout_loss=nn.CrossEntropyLoss(),
        train_dataloader=loader,
        test_dataloader=loader,
    )

    trainer.train()
    p = trainer.metrics_collector.train_metric_collector.get_metrics()
    p
