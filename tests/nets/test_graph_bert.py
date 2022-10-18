import dgl
import torch
from dgl.dataloading import GraphDataLoader

from lib.graph_bert.layers.attention_blocks.base import MultiHeadAttentionLayerConfig
from lib.graph_bert.layers.blocks.branch import BranchFFNConfig
from lib.graph_bert.layers.blocks.fully_connected import FullyConnectedConfig
from lib.graph_bert.layers.blocks.graph_transformer import GraphTransformerLayerConfig
from lib.graph_bert.layers.config.config_base import ReadOutConfig
from lib.graph_bert.layers.layers.norm import NormConfig
from lib.graph_bert.layers.layers.o_layer import OutputAttentionLayerConfig
from lib.graph_bert.layers.layers.readout import ReadOut
from lib.graph_bert.layers.mlp_readout_layer import MLPDefault, MLPConfig
from lib.graph_bert.nets.get_simple_config import get_bert_config_simple
from lib.graph_bert.nets.transform_block import GraphTransformBlockDefault, GraphBertConfig
from lib.graph_bert.nets.readout_mlp_net import ReadOutMlpDefault, ReadOutMlpConfig
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    FEATURE_COLUMN,
    MoleculeGraphBuilder,
)


def test_graph_bert(molecules):
    config = get_bert_config_simple()
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
    net = GraphTransformBlockDefault(config)
    h, e = net.forward(g, batch_h, batch_e)
    target = ReadOutMlpDefault(
        ReadOutMlpConfig(
            read_out_config=config.read_out_config,
            mlp_layer_config=config.mlp_layer_config,
        )
    )(g, h)

    assert list(target.shape) == [3, 768]
