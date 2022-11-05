import itertools

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn

from lib.graph_bert.layers.attention_blocks.base import MultiHeadAttentionLayerConfig
from lib.graph_bert.layers.blocks.branch import BranchFFNConfig
from lib.graph_bert.layers.blocks.fully_connected import FullyConnectedConfig
from lib.graph_bert.layers.blocks.graph_transformer import (
    GraphTransformerLayerDefault,
    GraphTransformerLayerConfig,
)
from lib.graph_bert.layers.layers.norm import NormConfig
from lib.graph_bert.layers.layers.o_layer import OutputAttentionLayerConfig
from lib.preprocessing.common import get_molecule_from_smile
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    FEATURE_COLUMN,
    MoleculeGraphBuilder,
)


def test_graph_transformer(molecules):

    in_dim = 16
    hidden_dim_attention = 16
    hidden_dim = 16
    e_dim = 16

    num_heads = 4

    fcc_hide = 1024
    batch = 3

    multy_head_attention_conf = MultiHeadAttentionLayerConfig(
        in_dim=in_dim, out_dim=hidden_dim_attention // num_heads, num_heads=num_heads
    )

    output_attention_config = OutputAttentionLayerConfig(
        in_dim=hidden_dim_attention,
        out_dim=hidden_dim,
    )

    pre_layer_norm = NormConfig(in_dim=hidden_dim)
    pre_batch_norm = NormConfig(in_dim=hidden_dim)

    fully_connected_config = FullyConnectedConfig(
        in_dim=hidden_dim, hidden_dim=fcc_hide, out_dim=hidden_dim
    )

    post_layer_norm = NormConfig(in_dim=hidden_dim)
    post_batch_norm = NormConfig(in_dim=hidden_dim)

    config_b = BranchFFNConfig(
        fully_connected_config=fully_connected_config,
        output_attention_config=output_attention_config,
        pre_layer_norm=pre_layer_norm,
        pre_batch_norm=pre_batch_norm,
        post_layer_norm=post_layer_norm,
        post_batch_norm=post_batch_norm,
    )
    graph_transformer_config = GraphTransformerLayerConfig(
        multy_head_attention_conf=multy_head_attention_conf,
        h_branch_config=config_b,
        e_branch_config=config_b,
    )

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

    h = embedding_h(batch_h).squeeze(1)
    e = embedding_e(batch_e).squeeze(1)

    graph_transformer = GraphTransformerLayerDefault(graph_transformer_config)
    target_h, target_e = graph_transformer.forward(g, h, e)
    target_h, target_e = graph_transformer.forward(g, target_h, target_e)
    target_h_size = list(target_h.shape)
    assert target_h_size == [num_atoms, post_batch_norm.in_dim]
