from typing import Any

from rdkit import Chem

from lib.preprocessing.common import get_v2i
from lib.preprocessing.models.bonds.features import BondProperty


class BondGlossary:
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    IN_RING = [True, False]
    IS_CONJUGATED = [True, False]
    STEREO_CHEMISTRY = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]

    MISSED = "UKNOWN"

    def __init__(self):
        self.bond_types2index = get_v2i(self.BOND_TYPES, self.MISSED)
        self.in_ring2index = get_v2i(self.IN_RING, False)
        self.is_conjugated2index = get_v2i(self.IS_CONJUGATED, False)
        self.stereo2index = get_v2i(self.STEREO_CHEMISTRY, self.MISSED)

    @property
    def as_dict(self):
        return {
            BondProperty.BOND_TYPE: self.bond_types2index,
            BondProperty.IN_RING: self.IN_RING,
            BondProperty.IS_CONJUGATED: self.is_conjugated2index,
            BondProperty.STEREO_CHEMISTRY: self.stereo2index,
        }

    def get_index_by_bond_property(
        self, value: Any, bond_property: BondProperty
    ) -> int:
        return self.as_dict[bond_property][value]
