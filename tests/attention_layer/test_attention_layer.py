import itertools

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn

from lib.graph_bert.layers.attention_blocks.base import MultiHeadAttentionLayerConfig
from lib.graph_bert.layers.attention_blocks.multy_head_attention import (
    MultiHeadAttentionLayerDefault,
)
from lib.preprocessing.common import get_molecule_from_smile
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import AtomGlossary, atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    MoleculeGraphBuilder,
    FEATURE_COLUMN,
)


def test_multy_head_attention_layer(molecules):

    batch = 30
    e_dim = 2

    config = MultiHeadAttentionLayerConfig()
    config.in_dim = e_dim
    config.out_dim = 2 * e_dim
    molecules *= 5

    atoms = list(
        itertools.chain.from_iterable(
            [
                [a.GetSymbol() for a in get_molecule_from_smile(smile).GetAtoms()]
                for smile in molecules
            ]
        )
    )
    num_atoms = len(atoms)

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
    for g, _ in loader:
        batch_h = g.ndata[FEATURE_COLUMN]
        batch_e = g.edata[FEATURE_COLUMN]

        break
    embedding_h = nn.Embedding(atom_glossary.get_num_atoms(), e_dim)
    embedding_e = nn.Embedding(100, e_dim)

    h = embedding_h(batch_h)
    e = embedding_e(batch_e)

    m_layer = MultiHeadAttentionLayerDefault(config)
    h_out, e_out = m_layer.forward(g, h, e)
    shape_h = list(h_out.shape)
    assert shape_h == [num_atoms, config.num_heads, config.out_dim]
