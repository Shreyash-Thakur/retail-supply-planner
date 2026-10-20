from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def test_load_config_returns_paths_and_seed():
    cfg = load_config()
    assert cfg["seed"] == 42
    for directory in cfg["data"].values():
        assert isinstance(directory, Path)
        assert directory.is_absolute()
        assert directory.is_relative_to(PROJECT_ROOT)


def test_load_config_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "raw"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"seed: 1\ndata:\n  raw_dir: '{target.as_posix()}'\n", encoding="utf-8")

    cfg = load_config(config_file)

    assert cfg["data"]["raw_dir"] == target
    assert target.is_dir()
