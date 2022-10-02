from enum import Enum

from lib.preprocessing.models.atom.atom_attributes import AtomAttributes


class AtomProperty(str, Enum):
    ATOM_TYPE = "atom_type"
    DEGREE = "degree"
    FORMAL_CHARGE = "formalCharge"
    HYBRIDIZATION = "hybridization"
    IN_RING = "inRing"
    IS_AROMATIC = "isAromatic"
    MASS = "mass"
    ATOMIC_NUM = "atomicNum"
    CHIRAL_TAG = "chiralTag"
    TOTAL_NUM_HS = "totalNumHs"


class AtomFeatures:
    def __init__(self, atom: AtomAttributes):
        self.atom = atom

    @property
    def as_dict(self):
        return {
            AtomProperty.ATOM_TYPE: self.atom.GetSymbol,
            AtomProperty.DEGREE: self.atom.GetDegree,
            AtomProperty.FORMAL_CHARGE: self.atom.GetFormalCharge,
            AtomProperty.HYBRIDIZATION: self.atom.GetHybridization,
            AtomProperty.IN_RING: self.atom.IsInRing,
            AtomProperty.IS_AROMATIC: self.atom.GetIsAromatic,
            AtomProperty.MASS: self.atom.GetMass,
            AtomProperty.ATOMIC_NUM: self.atom.GetAtomicNum,
            AtomProperty.CHIRAL_TAG: self.atom.GetChiralTag,
            AtomProperty.TOTAL_NUM_HS: self.atom.GetTotalNumHs,
        }
