#!/usr/bin/env python3
"""Benchmark the LIF backends (each in its own process) against the Brian2 reference.

    python scripts/benchmark.py [--duration 1.0] [--backends numba numpy torch]

Writes results/benchmark/benchmark.md. The Brian2 numbers come from
scripts/benchmark_brian2.py (run inside a Brian2 venv), if present.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORKER = r'''
import sys, time, json, numpy as np
sys.path.insert(0, %(root)r)
from flyhamlet.config import load_config
from flyhamlet.data import load_connectome
from flyhamlet.sim import LIFNetwork
cfg = load_config(); c = load_connectome(cfg, verbose=False)
present = set(c.root_ids.tolist())
ids = [i for i in cfg["validation"]["sugar_grn_ids"] if i in present]
net = LIFNetwork(c, cfg["sim"], seed=1, backend=%(backend)r)
net.inject_poisson({"ids": ids}, 150.0)
net.step()
n = int(round(%(dur)r * 1e3 / net.dt))
t0 = time.perf_counter(); rec = net.run(n_steps=n); dt = time.perf_counter() - t0
r = rec.rates_hz(c.n); mn9 = net.resolve({"ids": cfg["validation"]["mn9_ids"]})
h = hash((rec.steps.tobytes(), rec.neurons.tobytes()))
print(json.dumps({"backend": net.backend, "wall_s": dt, "ms_per_step": dt / n * 1e3, "x_realtime": dt / %(dur)r,
                  "spikes": int(len(rec.neurons)), "active": int((r > 0).sum()), "mn9_hz": r[mn9].tolist(), "spike_hash": h}))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--backends", nargs="+", default=["numba", "numpy", "torch"])
    a = ap.parse_args()
    rows = []
    for b in a.backends:
        code = WORKER % {"root": str(ROOT), "backend": b, "dur": a.duration}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": "0", **__import__("os").environ})
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr); rows.append({"backend": b, "error": out.stderr[-300:]}); continue
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1])
    lines = ["# Backend benchmark", "",
             f"Full v783 connectome (139,255 neurons, 15.09M connections), dt = 0.1 ms, 20 sugar GRNs driven at 150 Hz, {a.duration} s simulated, single process.", "",
             "| backend | wall (s) | ms / step | x real time | spikes | active neurons | MN9 L/R (Hz) | identical spikes |", "|---|---|---|---|---|---|---|---|"]
    ref = next((r.get("spike_hash") for r in rows if "spike_hash" in r), None)
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['backend']} | error | | | | | | |"); continue
        lines.append(f"| {r['backend']} | {r['wall_s']:.1f} | {r['ms_per_step']:.3f} | {r['x_realtime']:.1f}x | {r['spikes']} | {r['active']} | {r['mn9_hz'][0]:.0f}/{r['mn9_hz'][1]:.0f} | {'yes' if r['spike_hash']==ref else 'no'} |")
    bt = ROOT / "results/benchmark/brian2/brian2_timing.json"
    if bt.exists():
        t = json.load(open(bt))
        per = {k: v / 2 for k, v in t["timing_s_per_2_trials"].items()}
        lines += ["", f"Brian2 {t['brian2']} reference (Shiu et al. model.py, cython codegen, 1 process, includes network build per trial):", ""]
        lines += [f"- {k} Hz drive: {v:.0f} s per 1 s trial ({v:.0f}x real time)" for k, v in per.items()]
    lines += ["", "No GPU was available in this environment; the torch backend was timed on CPU only.",
              "Run one backend per process: numba's and torch's thread pools contend badly when mixed in one process."]
    (ROOT / "results/benchmark").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/benchmark/benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
