from typing import List

from lib.preprocessing.models.atom.atom_attributes import AtomAttributes
from lib.preprocessing.models.bonds.bonds_attributes import BondAttributes


class Molecule:
    def GetBondBetweenAtoms(self):
        pass

    def GetAtoms(self) -> List[AtomAttributes]:
        pass

    def GetNumAtoms(self) -> int:
        pass

    def GetNumBonds(self) -> int:
        pass

    def GetBonds(self) -> List[BondAttributes]:
        pass
