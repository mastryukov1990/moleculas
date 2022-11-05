import dgl
import torch
from dgl.dataloading import GraphDataLoader

from lib.graph_bert.nets.config import cfg
from lib.graph_bert.nets.transform_block import (
    GraphTransformBlockDefault,
)
from lib.graph_bert.nets.readout_mlp_net import ReadOutMlpDefault
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    FEATURE_COLUMN,
    MoleculeGraphBuilder,
)


def test_graph_bert(molecules):
    # config = get_bert_config_simple()
    batch = 10

    molecule_graph_builder = [
        MoleculeGraphBuilder.from_smile(
            molecule, atom_glossary, bond_glossary
        ).get_graph()
        for molecule in molecules
    ]
    labels = [1 for _ in range(len(molecules))]
    dataset = MoleculesDataset(molecule_graph_builder, labels)
    loader = GraphDataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
    )
    g = dgl.DGLHeteroGraph()
    batch_h = torch.Tensor()
    batch_e = torch.Tensor()
    for g, _ in loader:
        batch_h = g.ndata[FEATURE_COLUMN]
        batch_e = g.edata[FEATURE_COLUMN]

        break
    batch_h = batch_h.squeeze(1)
    batch_e = batch_e.squeeze(1)
    net = GraphTransformBlockDefault(cfg.transformer_block_config)
    h, e = net.forward(g, batch_h, batch_e)
    target = ReadOutMlpDefault(cfg.readout_config).forward(g, h)

    assert list(target.shape) == [3, cfg.readout_config.mlp_layer_config.out_dim]
