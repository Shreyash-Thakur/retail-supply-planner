"""Load config.yaml and resolve data directories."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict:
    """Return the parsed config with every `data` entry as an absolute Path.

    Relative paths (the config file itself and the data directories) are
    resolved against the project root, so this works from any working
    directory. Missing data directories are created.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg["data"].items():
        directory = Path(value)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        directory.mkdir(parents=True, exist_ok=True)
        cfg["data"][key] = directory
    return cfg
