"""Experiment 2: tap spikes from a neuron set and stream inter-spike intervals to disk.

The tap subscribes to ``n_neurons`` neurons sampled reproducibly (seeded) from
central-brain neurons that are not part of any wired sensory or motor pathway:
the looming input types, the descending neurons read out by the arena, and all
of their direct synaptic partners are excluded.  The sampled ID list is dumped
to ``<out_path>_neurons.csv``.

Binary format (little endian): header ``b"FHISI1"``, uint32 n_neurons, float64 dt_ms,
then a stream of records ``(uint16 neuron_slot, uint32 isi_steps)`` in the order
the spikes occur.  Read with :func:`read_isi_file`.

The randomness of the ISIs comes from the Poisson background drive and the
stochastic arena inputs, not from the (deterministic) wiring.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"FHISI1"
REC = np.dtype([("neuron", "<u2"), ("isi", "<u4")])


class EntropyTap:
    def __init__(self, indices: np.ndarray, root_ids: np.ndarray, out_path: str | Path, dt_ms: float,
                 background: dict | None = None, buffer: int = 1 << 16):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.root_ids = np.asarray(root_ids, dtype=np.int64)
        self.out_path = Path(out_path)
        self.dt_ms = float(dt_ms)
        self.background = background or {}
        self.slot = None            # dense index -> slot (or -1)
        self.last = np.full(len(self.indices), -1, dtype=np.int64)
        self.buf = np.zeros(buffer, dtype=REC); self.nbuf = 0
        self.n_isi = 0
        self.f = None
        self.sub = None

    # ----- construction from config
    @classmethod
    def from_config(cls, c, cfg: dict, fly_id: int = 0):
        tcfg = cfg["entropy_tap"]
        rng = np.random.default_rng(int(tcfg["seed"]))
        pool = c.super_class_mask(tcfg["pool_super_classes"])
        # exclude wired sensory/motor types and their direct partners
        from .sim import resolve_target
        a = cfg["arena"]
        wired_types = list(a["looming"]["types"]) + list(a["motor"]["turn_types"]) + list(a["motor"]["forward_types"]) \
            + list(a["motor"]["backward_types"]) + ["Giant_Fiber"]
        wired = np.concatenate([resolve_target(c, t) for t in wired_types])
        W = c.W
        partners = np.unique(np.concatenate([W[wired].indices, W[:, wired].tocsc().indices if False else W.T[wired].indices]))
        excl = np.zeros(c.n, dtype=bool); excl[wired] = True; excl[partners] = True
        cand = np.flatnonzero(pool & ~excl)
        idx = np.sort(rng.choice(cand, size=min(int(tcfg["n_neurons"]), len(cand)), replace=False))
        out = Path(str(tcfg["out_path"]))
        if not out.is_absolute():
            out = Path(cfg["_root"]) / out
        out = out.with_name(f"{out.stem}_fly{fly_id:02d}{out.suffix}")
        tap = cls(idx, c.root_ids[idx], out, cfg["sim"]["dt_ms"], tcfg.get("background"))
        tap.dump_neurons(c)
        print(f"EntropyTap: {len(idx)} neurons from a pool of {len(cand)} (excluded {int(excl.sum())} wired neurons/partners) -> {out}", file=sys.stderr)
        return tap

    def dump_neurons(self, c):
        import pandas as pd
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        d = c.describe(self.indices).copy(); d.insert(0, "slot", np.arange(len(self.indices)))
        d.to_csv(self.out_path.with_name(self.out_path.stem + "_neurons.csv"), index=False)

    # ----- wiring into a network
    def attach(self, net):
        self.slot = np.full(net.n, -1, dtype=np.int64)
        self.slot[self.indices] = np.arange(len(self.indices))
        self.sub = net.subscribe({"index": self.indices}, "entropy_tap")
        bg = self.background
        if bg and bg.get("enabled"):
            tgt = {"super_class": list(bg["super_classes"])} if bg.get("super_classes") else None
            net.set_background(float(bg["rate_hz"]), float(bg["weight_mV"]), tgt)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.out_path, "wb")
        self.f.write(MAGIC + struct.pack("<I", len(self.indices)) + struct.pack("<d", self.dt_ms))

    def on_step(self, step: int, spikes: np.ndarray):
        s = self.sub.last
        if len(s) == 0:
            return
        slots = self.slot[s]
        prev = self.last[slots]
        ok = prev >= 0
        k = int(ok.sum())
        if k:
            if self.nbuf + k > len(self.buf):
                self.flush()
            self.buf["neuron"][self.nbuf:self.nbuf + k] = slots[ok]
            self.buf["isi"][self.nbuf:self.nbuf + k] = step - prev[ok]
            self.nbuf += k; self.n_isi += k
        self.last[slots] = step

    def flush(self):
        if self.f is not None and self.nbuf:
            self.f.write(self.buf[:self.nbuf].tobytes()); self.nbuf = 0

    def close(self):
        self.flush()
        if self.f is not None:
            self.f.close(); self.f = None
        print(f"EntropyTap: wrote {self.n_isi} ISIs to {self.out_path}", file=sys.stderr)


def read_isi_file(path: str | Path):
    """Return (neuron_slots uint16 array, isi_steps uint32 array, n_neurons, dt_ms)."""
    b = Path(path).read_bytes()
    assert b[:6] == MAGIC, "not a FlyHamlet ISI file"
    n = struct.unpack("<I", b[6:10])[0]; dt = struct.unpack("<d", b[10:18])[0]
    rec = np.frombuffer(b[18:], dtype=REC)
    return rec["neuron"], rec["isi"], n, dt
