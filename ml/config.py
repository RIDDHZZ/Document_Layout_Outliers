"""Configuration loader."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def data_dir(cfg: dict) -> Path:
    return Path(cfg["_root"]) / cfg["paths"]["data_dir"]
