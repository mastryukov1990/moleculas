import torch
from hydra.core.config_store import ConfigStore
from torch import nn

from lib.graph_bert.layers.blocks.fully_connected import (
    FullyConnectedBlockBase,
    FullyConnectedConfig,
    FullyConnectedLeakyLayer,
)
from lib.graph_bert.layers.config.config_base import *
from lib.graph_bert.layers.layers.add import (
    AddLayerBase,
    SumAddLayer,
)
from lib.graph_bert.layers.layers.norm import (
    NormBase,
    NormConfig,
    LayerNorm,
    BatchNorm,
    NormConfigBlock,
    NormConfigName,
)
from lib.graph_bert.layers.layers.o_layer import (
    OutputAttentionLayerBase,
    OutputAttentionLayer,
    OutputAttentionLayerConfig,
)
from lib.logger import Logger

logger = Logger(__name__)


BlockBranchConfigGroup = "block_branch_config_group"
BlockBranchConfigName = "block_branch_config_name"


@dataclass
class BlockBranchConfig:
    fully_connected_config: FullyConnectedConfig = FullyConnectedConfig()
    output_attention_config: OutputAttentionLayerConfig = OutputAttentionLayerConfig()

    pre_layer_norm: NormConfig = NormConfig()
    pre_batch_norm: NormConfig = NormConfig()

    post_layer_norm: NormConfig = NormConfig()
    post_batch_norm: NormConfig = NormConfig()


@dataclass
class BranchFFNConfig(
    BlockBranchConfig,
    PreAddLayerConfig,
    PostAddLayerConfig,
):
    pass


class BranchFFNBase(nn.Module, metaclass=ABCMeta):
    ADD_LAYER = AddLayerBase
    OUTPUT_ATTENTION_LAYER = OutputAttentionLayerBase
    LAYER_NORM = NormBase
    BATCH_NORM = NormBase
    FULLY_CONNECTED_BLOCK = FullyConnectedBlockBase

    def __init__(self, config: BranchFFNConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor, x_res: torch.Tensor):
        pass


class BranchFFNAttention(BranchFFNBase):
    def __init__(self, config: BranchFFNConfig):
        super().__init__(config)

        self.output_attention_layer: OutputAttentionLayerBase = (
            self.OUTPUT_ATTENTION_LAYER(config.output_attention_config)
        )

        self.pre_add_layer = self.ADD_LAYER if config.pre_add_layer else None
        self.post_add_layer = self.ADD_LAYER if config.post_add_layer else None

        self.pre_layer_norm: NormBase = (
            self.LAYER_NORM(config.pre_layer_norm) if config.pre_layer_norm else None
        )
        self.pre_batch_norm: NormBase = (
            self.BATCH_NORM(config.pre_batch_norm) if config.pre_batch_norm else None
        )

        self.fully_connected: FullyConnectedBlockBase = self.FULLY_CONNECTED_BLOCK(
            config.fully_connected_config
        )

        self.post_layer_norm: NormBase = (
            self.LAYER_NORM(config.post_layer_norm) if config.post_layer_norm else None
        )

        self.post_batch_norm: NormBase = (
            self.BATCH_NORM(config.post_batch_norm) if config.post_batch_norm else None
        )

    def forward(self, x: torch.Tensor, x_res: torch.Tensor):
        logger.info(f"[{__name__}] h_attn_out = {x.shape} ")
        x = self.output_attention_layer.forward(x)

        if self.pre_add_layer:
            x = self.pre_add_layer.forward(x, x_res)

        if self.pre_layer_norm:
            x = self.pre_layer_norm.forward(x)

        if self.pre_batch_norm:
            x = self.pre_batch_norm.forward(x)

        x_post = torch.Tensor()

        if self.post_add_layer:
            x_post = x

        if self.fully_connected:
            x = self.fully_connected.forward(x)

        if self.post_add_layer:
            x = self.post_add_layer.forward(x, x_post)

        if self.post_layer_norm:
            x = self.post_layer_norm.forward(x)

        if self.post_batch_norm:
            x = self.post_batch_norm.forward(x)

        return x


class BranchFFNAttentionDefault(BranchFFNAttention):
    ADD_LAYER = SumAddLayer
    OUTPUT_ATTENTION_LAYER = OutputAttentionLayer
    LAYER_NORM = LayerNorm
    BATCH_NORM = BatchNorm
    FULLY_CONNECTED_BLOCK = FullyConnectedLeakyLayer


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=BlockBranchConfigGroup,
        name=BlockBranchConfigName,
        node=BlockBranchConfig,
    )
