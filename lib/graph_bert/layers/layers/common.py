from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from lib.graph_bert.layers.config.block_configs import ComposeInBlock
from lib.graph_bert.layers.layers.linear_layer import (
    LinearLayerConfig as llc,
    register_configs as rc_llc,
)
from lib.graph_bert.layers.layers.norm import (
    NormConfig as nc,
    register_configs as rc_nc,
)
from lib.graph_bert.layers.layers.o_layer import (
    OutputAttentionLayerConfig as oalc,
    register_configs as rc_oalc,
)
from lib.graph_bert.layers.layers.readout import (
    ReadOutConfig as roc,
    register_configs as rc_roc,
)


@dataclass
class LayersConfig:
    linear_layer: llc = llc()
    norm_layer: nc = nc()
    output_attention: oalc = oalc()
    readout: roc = roc()


cs = ConfigStore.instance()
cs.store(name="layers_config", node=LayersConfig)

rc_llc()
rc_nc()
rc_oalc()
# rc_roc()


@hydra.main(
    version_base=None,
    config_name="layers_config",
    # config_path="conf",
)
def my_app(cfg: LayersConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # with open("conf/config_bert.yaml", "w") as f:
    #     OmegaConf.save(cfg, f)


if __name__ == "__main__":
    my_app()
