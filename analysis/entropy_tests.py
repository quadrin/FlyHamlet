#!/usr/bin/env python3
"""NIST SP 800-90B entropy estimates on raw ISIs, plus dieharder-ready bitstreams.

* Most Common Value estimate (90B 6.3.1) on the raw ISI symbols and on their bit expansion.
* Collision estimate (90B 6.3.2; defined for binary sequences) on the bit expansion.
* Output: ``raw.bin`` (uint32 ISIs) and ``whitened.bin`` (SHA-256 stream) as raw 32-bit
  little-endian words for ``dieharder -g 201 -f <file>``-style ASCII input or ``-g 200`` binary
  (``--ascii`` writes the ASCII ``type: d`` format).

    python analysis/entropy_tests.py results/entropy/isi_fly*.bin --out results/entropy_exp2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- 90B 6.3.1
def most_common_value(samples: np.ndarray) -> dict:
    """Min-entropy upper bound from the most common value (99% upper confidence)."""
    s = np.asarray(samples)
    n = len(s)
    vals, counts = np.unique(s, return_counts=True)
    p_hat = counts.max() / n
    p_u = min(1.0, p_hat + 2.576 * math.sqrt(p_hat * (1 - p_hat) / max(1, n - 1)))
    return dict(n=int(n), alphabet=int(len(vals)), mode=vals[counts.argmax()].item(), p_hat=float(p_hat), p_upper=float(p_u),
                min_entropy_bits=float(-math.log2(p_u)))


# --------------------------------------------------------------------------- 90B 6.3.2
def _collision_expected_mean(p: float) -> float:
    q = 1.0 - p
    F = q + 2 * q ** 2 + 2 * q ** 3           # F(q) with F(1/z) = Gamma(3, z) z^-3 e^z
    return p * q ** -2 * (1 + 0.5 * (1 / p - 1 / q)) * F - p * q ** -1 * 0.5 * (1 / p - 1 / q)


def collision_estimate(bits: np.ndarray) -> dict:
    """Collision estimate for a binary sequence (SP 800-90B section 6.3.2)."""
    b = np.asarray(bits, dtype=np.uint8)
    n = len(b)
    times = []
    i = 0
    while i + 1 < n:
        if b[i] == b[i + 1]:
            times.append(2); i += 2
        elif i + 2 < n:
            times.append(3); i += 3      # third sample must repeat one of the first two
        else:
            break
    t = np.asarray(times, dtype=float)
    v = len(t)
    if v < 2:
        return dict(n=int(n), v=int(v), min_entropy_bits=float("nan"))
    xbar = t.mean(); sd = t.std(ddof=1)
    xprime = xbar - 2.576 * sd / math.sqrt(v)
    if xprime >= _collision_expected_mean(0.5):
        p = 0.5
    else:
        lo, hi = 0.5, 1.0 - 1e-12
        for _ in range(200):           # binary search: expected mean decreases with p
            mid = 0.5 * (lo + hi)
            if _collision_expected_mean(mid) > xprime:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
    return dict(n=int(n), v=int(v), mean_collision_time=float(xbar), x_prime=float(xprime), p=float(p),
                min_entropy_bits=float(-math.log2(p)))


def to_bits(samples: np.ndarray, width: int | None = None) -> tuple[np.ndarray, int]:
    """Fixed-width big-endian bit expansion of unsigned samples (width = bits needed for the max)."""
    s = np.asarray(samples, dtype=np.uint64)
    if width is None:
        width = max(1, int(s.max()).bit_length())
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((s[:, None] >> shifts) & np.uint64(1)).astype(np.uint8).ravel(), width


# --------------------------------------------------------------------------- dieharder output
def write_dieharder(words: np.ndarray, path: Path, ascii_fmt: bool = False):
    w = np.ascontiguousarray(words, dtype="<u4")
    if ascii_fmt:
        with open(path, "w") as f:
            f.write("#==================================================================\n# generator FlyHamlet\n")
            f.write("#==================================================================\ntype: d\ncount: %d\nnumbit: 32\n" % len(w))
            f.write("\n".join(str(int(x)) for x in w) + "\n")
    else:
        path.write_bytes(w.tobytes())


def main(argv=None):
    from flyhamlet.entropy_tap import read_isi_file
    from analysis.whiten import hash_batches
    ap = argparse.ArgumentParser()
    ap.add_argument("isi_files", nargs="+")
    ap.add_argument("--out", default="results/entropy_exp2")
    ap.add_argument("--ascii", action="store_true", help="dieharder ASCII (-g 202) instead of raw binary (-g 201)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    isis = np.concatenate([read_isi_file(f)[1] for f in a.isi_files]).astype(np.uint32)
    res = {"n_isi": int(len(isis))}
    res["mcv_raw_isi"] = most_common_value(isis)
    bits, width = to_bits(isis)
    res["bit_width"] = width
    res["mcv_isi_bits"] = most_common_value(bits)
    res["collision_isi_bits"] = collision_estimate(bits)
    low8 = (isis & 0xFF).astype(np.uint8)
    res["mcv_isi_low_byte"] = most_common_value(low8)
    lb, _ = to_bits(low8, 8)
    res["collision_isi_low_byte_bits"] = collision_estimate(lb)
    h = hash_batches(isis)
    hw = np.frombuffer(h[: len(h) // 4 * 4], dtype="<u4")
    res["mcv_whitened_bytes"] = most_common_value(np.frombuffer(h, dtype=np.uint8))
    hb, _ = to_bits(np.frombuffer(h, dtype=np.uint8), 8)
    res["collision_whitened_bits"] = collision_estimate(hb)
    write_dieharder(isis, out / ("raw.txt" if a.ascii else "raw.bin"), a.ascii)
    write_dieharder(hw, out / ("whitened.txt" if a.ascii else "whitened.bin"), a.ascii)
    json.dump(res, open(out / "entropy_tests.json", "w"), indent=1)
    L = ["# Experiment 2: entropy estimates (NIST SP 800-90B)", "", f"{len(isis):,} raw ISIs (timesteps) from {len(a.isi_files)} file(s).", "",
         "| data | estimator | min-entropy (bits/sample) | notes |", "|---|---|---|---|",
         f"| raw ISI symbols | MCV | {res['mcv_raw_isi']['min_entropy_bits']:.3f} | alphabet {res['mcv_raw_isi']['alphabet']:,}, mode {res['mcv_raw_isi']['mode']} (p={res['mcv_raw_isi']['p_hat']:.4f}) |",
         f"| raw ISI bits ({width}-bit) | MCV | {res['mcv_isi_bits']['min_entropy_bits']:.3f} | per bit |",
         f"| raw ISI bits ({width}-bit) | collision | {res['collision_isi_bits']['min_entropy_bits']:.3f} | per bit, p={res['collision_isi_bits']['p']:.4f} |",
         f"| raw ISI low byte | MCV | {res['mcv_isi_low_byte']['min_entropy_bits']:.3f} | per byte (max 8) |",
         f"| raw ISI low byte bits | collision | {res['collision_isi_low_byte_bits']['min_entropy_bits']:.3f} | per bit |",
         f"| whitened (SHA-256) bytes | MCV | {res['mcv_whitened_bytes']['min_entropy_bits']:.3f} | per byte (max 8) |",
         f"| whitened (SHA-256) bits | collision | {res['collision_whitened_bits']['min_entropy_bits']:.3f} | per bit |", "",
         f"dieharder input written to `{out}/raw.*` (uint32 ISIs) and `{out}/whitened.*` (SHA-256 stream): "
         f"`dieharder -a -g {'202' if a.ascii else '201'} -f <file>`.", ""]
    (out / "report.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
