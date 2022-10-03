from typing import List, Any, Dict
from rdkit import Chem

from lib.preprocessing.models.molecule import Molecule


class MissedDict(dict):
    def __init__(self, d, missed_value=0):
        super(MissedDict, self).__init__(d)
        self.missed_atom_value = missed_value

    def __missing__(self, key):
        return self.missed_atom_value


def get_v2i(values: List[Any], missed_value: Any) -> Dict[int, Any]:
    return MissedDict({x[1]: x[0] for x in enumerate(values)}, missed_value)


def get_molecule_from_smile(smile: str) -> Molecule:
    return Chem.MolFromSmiles(smile)
