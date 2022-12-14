from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor

from lib.preprocessing.models.atom.directory import AtomGlossary
from lib.preprocessing.models.bonds.directory import BondGlossary


class Graph:
    ndata: Dict[str, Tensor]
    edata: Dict[str, Tensor]


class GraphBuilder(ABC):
    @abstractmethod
    def get_graph(self) -> Graph:
        pass

    @abstractmethod
    def from_smile(
        cls,
        smile: str,
        atom_glossary: AtomGlossary,
        bond_glossary: BondGlossary,
    ):
        pass
