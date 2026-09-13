"""Run N independent flies (separate seeds) in parallel processes.

    python -m flyhamlet.run_flies [--n-flies 4] [--duration 60] [--seed 42] [--procs 4] [--live] [--tap]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

from .config import load_config, resolve
from .data import load_connectome

_C = None
_CFG = None
_ARGS = None


def _run_one(i: int):
    from .arena import FlyArena
    cfg = _CFG
    seed = int(cfg["arena"]["seed"]) + i
    taps = []
    if _ARGS.tap:
        from .entropy_tap import EntropyTap
        taps.append(EntropyTap.from_config(_C, cfg, fly_id=i))
    a = FlyArena(_C, cfg, seed=seed, backend=_ARGS.backend, taps=taps)
    cb = None
    if _ARGS.live and i == 0:
        from .viewer import LiveView
        cb = LiveView(a)
    dur = _ARGS.duration or (cfg["entropy_tap"]["duration_s"] if _ARGS.tap else cfg["arena"]["duration_s"])
    a.run(float(dur), on_control=cb)
    summ = a.save(resolve(cfg, cfg["arena"]["log_dir"]), i, seed)
    print(f"fly {i:2d} seed {seed}: {summ['keystrokes']} keystrokes ({summ['keystrokes_per_s']:.2f}/s), "
          f"{summ['spikes']} spikes, wall contact {summ['wall_contact_fraction']:.1%}: {summ['text'][:60]!r}", file=sys.stderr, flush=True)
    return summ


def main(argv=None):
    global _C, _CFG, _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-flies", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--live", action="store_true", help="matplotlib live view of fly 0 (forces --procs 1)")
    ap.add_argument("--tap", action="store_true", help="attach the EntropyTap (Phase 3, experiment 2)")
    ap.add_argument("--log-dir", default=None, help="override arena.log_dir")
    _ARGS = ap.parse_args(argv)
    _CFG = load_config(_ARGS.config)
    if _ARGS.seed is not None:
        _CFG["arena"]["seed"] = _ARGS.seed
    if _ARGS.log_dir is not None:
        _CFG["arena"]["log_dir"] = _ARGS.log_dir
    _C = load_connectome(_CFG, verbose=False)
    n = _ARGS.n_flies or int(_CFG["arena"]["n_flies"])
    out = resolve(_CFG, _CFG["arena"]["log_dir"]); out.mkdir(parents=True, exist_ok=True)
    if _ARGS.live or _ARGS.procs <= 1 or n == 1:
        res = [_run_one(i) for i in range(n)]
    else:
        with mp.get_context("fork").Pool(min(_ARGS.procs, n)) as pool:
            res = pool.map(_run_one, range(n), chunksize=1)
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    print(f"wrote {out}/fly*_keystrokes.csv for {n} flies; total keystrokes {sum(r['keystrokes'] for r in res)}")


if __name__ == "__main__":
    main()
