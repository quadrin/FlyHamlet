"""Single-YAML configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config. Relative paths inside it are resolved against the repo root."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict[str, Any], path: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else Path(cfg.get("_root", ROOT)) / p
