import hydra
from omegaconf import OmegaConf

from lib.graph_bert.layers.layers.common import LayersConfig


@hydra.main(
    version_base=None,
    config_name="config_bert",
    config_path="../../lib/graph_bert/layers/layers/conf",
)
def my_app(cfg: LayersConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # with open("conf/config_bert.yaml", "w") as f:
    #     OmegaConf.save(cfg, f)


if __name__ == "__main__":
    my_app()
