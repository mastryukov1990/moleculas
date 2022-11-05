from dataclasses import dataclass

import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from lib.graph_bert.nets.mask_classifier import MaskClassifierConfig
from lib.graph_bert.nets.readout_mlp_net import ReadOutMlpConfig
from lib.graph_bert.nets.transform_block import GraphBertTransformerConfig


@dataclass
class GraphBertConfig:
    is_classifier_mask_config: bool = True
    is_readout_config: bool = True
    classifier_mask_config: MaskClassifierConfig = MaskClassifierConfig()
    readout_config: ReadOutMlpConfig = ReadOutMlpConfig()
    transformer_block_config: GraphBertTransformerConfig = GraphBertTransformerConfig()


cs = ConfigStore.instance()
cs.store(name="config_bert", node=GraphBertConfig)


@hydra.main(version_base=None, config_name="config_bert")
def my_app(cfg: GraphBertConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    with open("config_bert/config_bert.yaml", "w") as f:
        OmegaConf.save(cfg, f)


def get_config_from_path() -> GraphBertConfig:
    initialize(config_path='config_bert', job_name="test_app")
    cfg = compose(
        config_name="config_bert",
    )
    return cfg


cfg = get_config_from_path()

if __name__ == "__main__":
    my_app()
