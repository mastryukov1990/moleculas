from enum import Enum
from typing import Dict

import attrs as attrs

from lib.preprocessing.models.atom.features import AtomProperty


class AtomAttributes:
    def GetIdx(self):
        pass

    def GetSymbol(self):
        pass

    def GetDegree(self):
        pass

    def GetFormalCharge(self):
        pass

    def GetHybridization(self):
        pass

    def IsInRing(self):
        pass

    def GetIsAromatic(self):
        pass

    def GetMass(self):
        pass

    def GetAtomicNum(self):
        pass

    def GetChiralTag(self):
        pass

    def GetTotalNumHs(self):
        pass
