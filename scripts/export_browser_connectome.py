#!/usr/bin/env python3
"""Export the complete configured FlyWire connectome for the browser worker.

Run from the repository root with ``python scripts/export_browser_connectome.py``.
The existing data loader downloads and caches the public v783 release tables.
There is no threshold, neuron selection, or behavioral recording in this export.
Each gzip stream contains one little-endian typed array, without a header.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyhamlet.config import load_config, resolve  # noqa: E402
from flyhamlet.data import load_connectome  # noqa: E402
from flyhamlet.typewriter import Typewriter  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, values: np.ndarray, dtype: str) -> dict:
    """Write deterministic gzip and verify the decompressed bytes before publishing."""
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    raw = memoryview(array).cast("B")
    expected = hashlib.sha256(raw).hexdigest()
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        # No original filename and a fixed mtime make repeated exports identical.
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0,
                           compresslevel=9) as compressed:
            compressed.write(raw)
    actual = hashlib.sha256()
    size = 0
    with gzip.open(temporary, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            actual.update(chunk)
            size += len(chunk)
    if actual.hexdigest() != expected or size != len(raw):
        raise RuntimeError(f"gzip round-trip failed for {path.name}")
    if temporary.stat().st_size >= 50_000_000:
        raise RuntimeError(f"{path.name} exceeds the 50 MB browser asset limit")
    temporary.replace(path)
    return {
        "dtype": dtype,
        "length": int(array.size),
        "bytes": path.stat().st_size,
        "rawBytes": len(raw),
        "sha256": sha256(path),
        "rawSha256": expected,
    }


def export(config_path: str | None, output: Path) -> dict:
    cfg = load_config(config_path)
    c = load_connectome(cfg)
    if cfg["data"]["files"]["connections"] != "connections_no_threshold.csv.gz":
        raise ValueError("Browser export requires the full, unthresholded connection table")
    if c.n >= 2**32 or c.W.nnz >= 2**32:
        raise ValueError("Connectome does not fit the browser's uint32 CSR format")
    output.mkdir(parents=True, exist_ok=True)

    # Match flyhamlet.sim.LIFNetwork exactly: float64 arithmetic, float32 storage.
    pre = np.repeat(np.arange(c.n), np.diff(c.W.indptr))
    weights = (c.W.data.astype(np.float64) * c.sign[pre]
               * float(cfg["sim"]["w_syn_mV"])).astype(np.float32)
    del pre
    files = {"indptr": "indptr.u32.gz", "indices": "indices.u32.gz",
             "weights": "weights.f32.gz"}
    integrity = {}
    for name, values, dtype in (("indptr", c.W.indptr, "<u4"),
                                ("indices", c.W.indices, "<u4"),
                                ("weights", weights, "<f4")):
        integrity[name] = write_array(output / files[name], values, dtype)
        print(f"{files[name]}: {integrity[name]['bytes']:,} compressed bytes", flush=True)

    arena, sim = cfg["arena"], dict(cfg["sim"])
    sim["rest_eps_mV"] = float(sim.get("rest_eps_mV", 1e-5))
    motor, looming = arena["motor"], arena["looming"]
    # Pandas nullable strings can yield an object mask; normalize missing sides.
    side = c.neurons["side"].fillna("").to_numpy(dtype=str)

    def motor_group(types: list[str], hemisphere: str) -> list[int]:
        # Same concatenate + side selection as FlyArena.add_group.
        indices = np.concatenate([c.neurons_of_type(t) for t in types])
        return indices[side[indices] == hemisphere].tolist()

    targets = {}
    for hemisphere, suffix in (("left", "L"), ("right", "R")):
        for label, field in (("turn", "turn_types"), ("fwd", "forward_types"),
                             ("bwd", "backward_types")):
            targets[f"{label}_{suffix}"] = motor_group(motor[field], hemisphere)
        targets[f"eye_{suffix}"] = np.unique(
            motor_group(looming["types"], hemisphere)).tolist()
    targets["GF"] = c.neurons_of_type("Giant_Fiber").tolist()
    if any(not values for values in targets.values()):
        raise ValueError("The configured sensory/motor targets must all be present")

    tw = arena["typewriter"]
    layout = Typewriter(arena["width_mm"], arena["height_mm"], tw["rows"],
                        tw["cols"], tw["alphabet"], tw["layout_seed"]).letters
    sources = []
    for name, filename in cfg["data"]["files"].items():
        source = resolve(cfg, cfg["data"]["raw_dir"]) / filename
        sources.append({"table": name,
                        "url": f"{cfg['data']['base_url'].rstrip('/')}/{filename}",
                        "bytes": source.stat().st_size, "sha256": sha256(source)})
    manifest = {
        "format": "flyhamlet-csr-v1",
        "n": c.n,
        "edgeCount": int(c.W.nnz),
        "synapseCount": int(c.W.data.sum()),
        "files": files,
        "integrity": integrity,
        "config": {"sim": sim, "arena": arena},
        "targets": targets,
        "layout": layout,
        "provenance": {
            "dataset": "FlyWire FAFB v783 public Codex release",
            "release": cfg["data"]["version"],
            "source": "https://codex.flywire.ai/",
            "sources": sources,
            "exporter": "scripts/export_browser_connectome.py",
            "ordering": "All neurons.csv root IDs sorted ascending; dense zero-based indices",
            "rootIdsSha256": hashlib.sha256(c.root_ids.astype("<i8").tobytes()).hexdigest(),
            "connectivity": "All connection rows between release neurons; duplicate pre/post pairs summed; no threshold or pruning",
            "weights": "float32(float64(syn_count) * presynaptic_sign * w_syn_mV)",
            "ntSign": cfg["data"]["nt_sign"],
            "unknownNtSign": cfg["data"]["unknown_nt_sign"],
            "targetRootIds": {name: [str(c.root_ids[i]) for i in indices]
                              for name, indices in targets.items()},
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {c.n:,} neurons, {c.W.nnz:,} connected pairs, "
          f"{manifest['synapseCount']:,} synapses.")
    print("Target counts:", {name: len(indices) for name, indices in targets.items()})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional configuration YAML")
    parser.add_argument("--output", type=Path, default=ROOT / "site/model")
    args = parser.parse_args()
    export(args.config, args.output)
