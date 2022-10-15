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
from lib.graph_bert.layers.mlp_readout_layer import MLP, MLPConfig
from lib.graph_bert.nets.graph_bert import GraphBertDefault, GraphBertConfig
from lib.preprocessing.dataset import MoleculesDataset
from lib.preprocessing.models.atom.glossary import atom_glossary
from lib.preprocessing.models.bonds.glossary import bond_glossary
from lib.preprocessing.models.molecul_graph_builder.dgl_graph import (
    FEATURE_COLUMN,
    MoleculeGraphBuilder,
)


def test_graph_bert(molecules):
    hidden_dim_attention = 768
    hidden_dim = 768
    fcc_hide = 10
    num_heads = 12
    out_dim = 768

    multy_head_attention_conf = MultiHeadAttentionLayerConfig(
        in_dim=hidden_dim, num_heads=num_heads, out_dim=hidden_dim // num_heads
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

    h_branch_config = BranchFFNConfig(
        fully_connected_config=fully_connected_config,
        output_attention_config=output_attention_config,
        pre_layer_norm=pre_layer_norm,
        pre_batch_norm=pre_batch_norm,
        post_layer_norm=post_layer_norm,
        post_batch_norm=post_batch_norm,
    )

    config = GraphBertConfig(
        graph_transformer_layer_config=GraphTransformerLayerConfig(
            multy_head_attention_conf=multy_head_attention_conf,
            h_branch_config=h_branch_config,
            e_branch_config=h_branch_config,
        ),
        graph_transformer_layer_config_out=GraphTransformerLayerConfig(
            multy_head_attention_conf=multy_head_attention_conf,
            h_branch_config=h_branch_config,
            e_branch_config=h_branch_config,
        ),
        read_out_config=ReadOutConfig(),
        mlp_layer_config=MLPConfig(in_dim=hidden_dim, out_dim=out_dim),
        num_transforms=12,
        pos_enc_dim=10,
    )
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
    net = GraphBertDefault(config)
    target = net.forward(g, batch_h, batch_e)

    assert list(target.shape) == [3, out_dim]
