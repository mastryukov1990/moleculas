from enum import Enum
from typing import Optional, List

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
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"


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


@attr.s
class HiddenDimConfig:
    hidden_dim: int = attr.ib(default=256)


@attr.s
class InDimConfig:
    in_dim: int = attr.ib(default=128)


@attr.s
class OutDimConfig:
    out_dim: int = attr.ib(default=128)


@attr.s
class OutDimConfig:
    out_dim: int = attr.ib(default=128)


@attr.s
class DropoutConfig:
    dropout: float = attr.ib(default=0.1)


@attr.s
class InFeatDropout:
    in_feat_dropout: float = attr.ib(default=0.1)


@attr.s
class NClassesConfig:
    n_classes: int = attr.ib(default=2)


@attr.s
class NumTransformsConfig:
    num_transforms: int = attr.ib(default=10)


@attr.s
class NumHeadsConfig:
    num_heads: int = attr.ib(default=4)


@attr.s
class ReadoutConfig:
    readout: Readout = attr.ib(default=Readout.MEAN)


@attr.s
class LayerNormConfig:
    layer_norm: bool = attr.ib(default=False)


@attr.s
class BatchNormConfig:
    batch_norm: int = attr.ib(default=True)


@attr.s
class ResidualConfig:
    residual: bool = attr.ib(default=True)


@attr.s
class DeviceConfig:
    device: str = attr.ib(default="cpu")


@attr.s
class PosEncDimConfig:
    pos_enc_dim: int = attr.ib(default=10)


@attr.s
class MaxWlRoleIndexConfig:
    max_wl_role_index: int = attr.ib(default=37)


@attr.s
class BiasConfig:
    bias: bool = attr.ib(default=True)


@attr.s
class ActivationConfig:
    activation: bool = attr.ib(default=True)


@attr.s
class PreAddLayerConfig:
    pre_add_layer: bool = attr.ib(default=True)


@attr.s
class PostAddLayerConfig:
    post_add_layer: bool = attr.ib(default=True)


@attr.s
class NumHiddenConfig:
    num_hidden: int = attr.ib(default=10)
