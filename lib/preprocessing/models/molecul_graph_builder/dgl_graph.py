from typing import List, Tuple

import dgl
import torch
from torch import Tensor


from lib.preprocessing.common import get_molecule_from_smile
from lib.preprocessing.models.atom.directory import AtomGlossary
from lib.preprocessing.models.atom.features import AtomProperty, AtomFeatures
from lib.preprocessing.models.bonds.directory import BondGlossary
from lib.preprocessing.models.bonds.features import BondProperty, BondFeatures
from lib.preprocessing.models.molecul_graph_builder.graph_base import GraphBuilder  # noqa: E501

FEATURE_COLUMN = "feat"


class MoleculeGraphBuilder(GraphBuilder):
    NODE_FEATURES: List[AtomProperty] = [AtomProperty.ATOM_TYPE]
    BOND_FEATURES: List[BondProperty] = [BondProperty.BOND_TYPE]

    def __init__(
        self,
        features_nodes: Tensor,
        features_edges: Tensor,
        edges: List[Tuple[int, int]],
    ):
        self.g = dgl.DGLGraph()
        self.g.add_nodes(len(features_nodes))
        self.g.ndata[FEATURE_COLUMN] = features_nodes

        for src, dst in edges:
            self.g.add_edges(src, dst)

        self.g.edata[FEATURE_COLUMN] = features_edges

    @classmethod
    def num_node_features(cls) -> int:
        return len(cls.NODE_FEATURES)

    @classmethod
    def num_bond_features(cls) -> int:
        return len(cls.BOND_FEATURES)

    def get_graph(self) -> dgl.DGLGraph:
        return self.g

    @classmethod
    def from_smile(
        cls,
        smile: str,
        atom_glossary: AtomGlossary,
        bond_glossary: BondGlossary,
    ):
        molecule = get_molecule_from_smile(smile)

        atoms = molecule.GetAtoms()

        node_feature_indexes = torch.zeros(
            (len(atoms), cls.num_node_features()),
        )

        for atom in molecule.GetAtoms():
            atom_features = AtomFeatures(atom)
            atom_index = atom.GetIdx()

            node_feature_indexes[atom_index, :] = torch.Tensor(
                [
                    atom_glossary.get_index_by_atom_property(
                        value=atom_features.as_dict[node_feature](),
                        atom_property=node_feature,
                    )
                    for node_feature in cls.NODE_FEATURES
                ]
            )

        bonds = molecule.GetBonds()

        bond_feature_indexes = torch.zeros(
            (len(bonds), cls.num_bond_features()),
        )
        bond_indexes: List[Tuple[int, int]] = []

        for k, bond in enumerate(bonds):
            bond_features = BondFeatures(bond)

            bond_feature_indexes[k, :] = torch.Tensor(
                [
                    bond_glossary.get_index_by_bond_property(
                        value=bond_features.as_dict[bond_feature](),
                        bond_property=bond_feature,
                    )
                    for bond_feature in cls.BOND_FEATURES
                ]
            )

            bond_indexes.append(
                (
                    bond.GetBeginAtom().GetIdx(),
                    bond.GetEndAtom().GetIdx(),
                )
            )

        return cls(
            node_feature_indexes,
            bond_feature_indexes,
            bond_indexes,
        )
