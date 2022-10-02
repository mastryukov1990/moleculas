from lib.preprocessing.models.atom.atom_attributes import AtomAttributes


class BondAttributes:
    def GetBondType(self):
        pass

    def GetIsConjugated(self):
        pass

    def IsInRing(self):
        pass

    def GetStereo(self):
        pass

    def GetBeginAtom(self) -> AtomAttributes:
        pass

    def GetEndAtom(self) -> AtomAttributes:
        pass
