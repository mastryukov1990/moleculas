from lib.graph_bert.layers.blocks.fully_connected import FullyConnectedConfig
from lib.graph_bert.layers.blocks.graph_transformer import GraphTransformerLayerConfig
from lib.graph_bert.layers.config.config_base import *


@attr.s
class NetConfig(
    FullyConnectedConfig,
    GraphTransformerLayerConfig,
    InFeatDropout,
    NClassesConfig,
    NumTransformsConfig,
    NumHeadsConfig,
    DeviceConfig,
    PosEncDimConfig,
    MaxWlRoleIndexConfig,
    Config,
):
    pass
