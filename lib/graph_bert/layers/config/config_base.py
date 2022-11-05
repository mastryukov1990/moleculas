import abc
import copy
from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict

import attr
from pydantic import BaseModel
import yaml
import json

from lib.logger import Logger

logger = Logger(__name__)

BASE_SECTION = "base"

BRANCH_FFN_SECTION = "branch_ffn"

MULTI_HEAD_ATTENTION_LAYER_SECTION = "multi_head_attention_layer"


class Readout(str, Enum):
    max = "max"
    mean = "mean"
    sum = "sum"


class Config:
    SECTIONS = []

    @classmethod
    def from_file(cls, filename: str, is_yaml: bool = False):
        if is_yaml:
            logger.info(f"Loading config for {cls} from yaml")
            with open(filename, "r") as f:
                data = yaml.safe_load(f)
        elif filename.startswith("{"):
            logger.info(f"Loading config for {cls} from json")
            data = json.loads(filename)
        else:
            logger.info(f"Loading config for {cls} from file {filename}")
            with open(filename, "r") as f:
                data = json.load(f)

        if cls.SECTIONS:
            logger.info(f"Using sections {cls.SECTIONS}")
            joined_sections_data = {}
            for section in cls.SECTIONS:
                joined_sections_data.update(data[section])
            data = joined_sections_data

        result = cls(**data)
        return result


class CopyConfig:
    def get_copy(self):
        return copy.copy(self)


class FromDictConfig:
    @classmethod
    def from_dict(cls, fields: Dict):
        return cls(**fields)


@dataclass
class HiddenDimConfig:
    hidden_dim: int = 256


@dataclass
class InDimConfig:
    in_dim: int = 128


@dataclass
class OutDimConfig:
    out_dim: int = 128


@dataclass
class DropoutConfig:
    dropout: float = 0.1


@dataclass
class InFeatDropout:
    in_feat_dropout: float = 0.1


@dataclass
class NClassesConfig:
    n_classes: int = 2


@dataclass
class NumTransformsConfig:
    num_transforms: int = 10


@dataclass
class NumHeadsConfig:
    num_heads: int = 4


@dataclass
class ReadOutConfig:
    readout: Readout = Readout.mean


@dataclass
class LayerNormConfig:
    layer_norm: bool = False


class BatchNormConfig(abc.ABC):
    batch_norm: bool = True


@dataclass
class ResidualConfig:
    residual: bool = True


@dataclass
class DeviceConfig:
    device: str = "cpu"


@dataclass
class PosEncDimConfig:
    pos_enc_dim: int = 10


@dataclass
class MaxWlRoleIndexConfig:
    max_wl_role_index: int = 37


@dataclass
class BiasConfig:
    bias: bool = True


@dataclass
class ActivationConfig:
    activation: bool = True


@dataclass
class PreAddLayerConfig:
    pre_add_layer: bool = True


@dataclass
class PostAddLayerConfig:
    post_add_layer: bool = True


@dataclass
class NumHiddenConfig:
    num_hidden: int = 10


@dataclass
class PosEncDim:
    pos_enc_dim: int = 10


@dataclass
class MaxWlRoleIndex:
    max_wl_role_index: int = 37


@dataclass
class NumAtomType:
    num_atom_type: int = 100


@dataclass
class NumBondType:
    num_bond_type: int = 100
