import tomllib
from pathlib import Path


def config_loader() -> dict[str, dict]:
    with Path("config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
        return config
