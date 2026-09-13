#!/usr/bin/env python3
"""Download the FlyWire v783 Codex tables into data/raw and build the cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flyhamlet.config import load_config          # noqa: E402
from flyhamlet.data import download, load_connectome  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    for k, p in download(cfg).items():
        print(f"{k:20s} {p}  ({p.stat().st_size/1e6:.1f} MB)")
    c = load_connectome(cfg, rebuild=True)
    print(c.summary())
