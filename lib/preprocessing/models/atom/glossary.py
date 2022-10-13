from typing import Any

from lib.preprocessing.common import get_v2i
from lib.preprocessing.models.atom.features import AtomProperty


class AtomGlossary:
    ATOM_TYPES = [
        "H",
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]
    ATOM_TYPES_MISSED = "Unknown"

    DEGREE = [0, 1, 2, 3, 4]
    DEGREE_MISSED = "MoreThanFour"

    CHARGE = [-3, -2, -1, 0, 1, 2, 3]
    CHARGE_MISSED = "Extreme"

    HYBRIDIZATION = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]
    HYBRIDIZATION_MISSED = "OTHER"

    IN_RING = [True, False]

    IS_AROMATIC = [True, False]

    CHIRAL_TAGS = [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
    ]
    CHIRAL_TAGS_MISSED = "CHI_OTHER"

    TOTAL_NUM_HS = [0, 1, 2, 3, 4]
    TOTAL_NUM_HS_MISSED = "MoreThanFour"

    def __init__(self):
        self.atom_type2index = get_v2i(self.ATOM_TYPES, self.ATOM_TYPES_MISSED)
        self.degree2index = get_v2i(self.DEGREE, self.DEGREE_MISSED)
        self.charge2index = get_v2i(self.CHARGE, self.CHARGE_MISSED)
        self.hybridization2index = get_v2i(
            self.HYBRIDIZATION, self.HYBRIDIZATION_MISSED
        )
        self.chiral_tags2index = get_v2i(self.CHIRAL_TAGS, self.CHIRAL_TAGS_MISSED)
        self.total_num_hs2index = get_v2i(self.TOTAL_NUM_HS, self.TOTAL_NUM_HS_MISSED)
        self.in_ring2index = get_v2i(self.IN_RING, False)
        self.is_aromatic2index = get_v2i(self.IS_AROMATIC, False)

    def get_charge_index(self, value: Any):
        return self.charge2index[value]

    def get_hybridization_index(self, value: Any):
        return self.hybridization2index[value]

    def get_in_ring_index(self, value: Any):
        return self.in_ring2index[value]

    def get_is_aromatic_index(self, value: Any):
        return self.is_aromatic2index[value]

    def get_chiral_tags_index(self, value: Any):
        return self.chiral_tags2index[value]

    def get_total_num_hs_index(self, value: Any):
        return self.total_num_hs2index[value]

    @property
    def as_dict(self):
        return {
            AtomProperty.ATOM_TYPE: self.atom_type2index,
            AtomProperty.DEGREE: self.degree2index,
            AtomProperty.FORMAL_CHARGE: self.charge2index,
            AtomProperty.HYBRIDIZATION: self.hybridization2index,
            AtomProperty.IN_RING: self.in_ring2index,
            AtomProperty.IS_AROMATIC: self.is_aromatic2index,
            AtomProperty.CHIRAL_TAG: self.chiral_tags2index,
            AtomProperty.TOTAL_NUM_HS: self.total_num_hs2index,
        }

    def get_index_by_atom_property(
        self, value: Any, atom_property: AtomProperty
    ) -> int:
        return self.as_dict[atom_property][value]

    def get_num_atoms(self):
        return len(self.ATOM_TYPES)


atom_glossary = AtomGlossary()
