from enum import Enum
from typing import Dict, Any

from lib.preprocessing.models.bonds.bonds_attributes import (
    BondAttributes,
)


class BondProperty(str, Enum):
    STEREO_CHEMISTRY = "atom_type"
    BOND_TYPE = "bond_type"
    IN_RING = "inRing"
    IS_CONJUGATED = "is_conjugated"


class BondFeatures:
    def __init__(self, bond: BondAttributes):
        self.bond = bond

    @property
    def as_dict(self) -> Dict[BondProperty, Any]:
        return {
            BondProperty.STEREO_CHEMISTRY: self.bond.GetStereo,
            BondProperty.BOND_TYPE: self.bond.GetBondType,
            BondProperty.IN_RING: self.bond.IsInRing,
            BondProperty.IS_CONJUGATED: self.bond.GetIsConjugated,
        }
