from typing import List, Any, Dict
from rdkit import Chem

from lib.preprocessing.models.molecule import Molecule


def get_v2i(values: List[Any]) -> Dict[int, Any]:
    return {x[1]: x[0] for x in enumerate(values)}


def get_molecule_from_smile(smile: str) -> Molecule:
    return Chem.MolFromSmiles(smile)
