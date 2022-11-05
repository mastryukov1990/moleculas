import hydra
from omegaconf import OmegaConf


def get_hydra_main(ConfigClass, config_name: str, config_path: str):
    @hydra.main(version_base=None, config_name=config_name, config_path=config_path)
    def my_app(cfg: config_name) -> None:
        print(OmegaConf.to_yaml(cfg))
        with open(f"{ConfigClass}.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    return my_app
