#!/usr/bin/env python3
"""Whiten raw inter-spike intervals into uniform keys in {a-z, space}.

Batches of 64 ISIs (uint32 timesteps, stream order) are hashed with SHA-256; each
output byte b becomes a key iff b < 243 (= 9 * 27, rejection sampling), key = b % 27.
Raw ISIs are never used mod 27.

    python analysis/whiten.py results/entropy/isi_fly00.bin [--out keys.txt] [--hash-bin whitened.bin]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
REJECT_ABOVE = 243          # largest multiple of 27 that fits in a byte
BATCH = 64


def hash_batches(isis: np.ndarray, batch: int = BATCH) -> bytes:
    """Concatenated SHA-256 digests of consecutive batches of ``batch`` ISIs (trailing partial batch dropped)."""
    isis = np.ascontiguousarray(isis, dtype="<u4")
    n = len(isis) // batch
    out = bytearray()
    for i in range(n):
        out += hashlib.sha256(isis[i * batch:(i + 1) * batch].tobytes()).digest()
    return bytes(out)


def bytes_to_keys(b: bytes, reject_above: int = REJECT_ABOVE) -> np.ndarray:
    """Rejection sampling: keep bytes < reject_above, map with mod 27. Returns key indices."""
    arr = np.frombuffer(b, dtype=np.uint8)
    keep = arr[arr < reject_above]
    return (keep % len(ALPHABET)).astype(np.int64)


def keys_to_text(keys: np.ndarray) -> str:
    return "".join(ALPHABET[k] for k in keys)


def whiten(isis: np.ndarray, batch: int = BATCH) -> tuple[str, bytes, dict]:
    h = hash_batches(isis, batch)
    keys = bytes_to_keys(h)
    stats = dict(n_isi=int(len(isis)), n_batches=len(h) // 32, hash_bytes=len(h), keys=int(len(keys)),
                 accept_fraction=float(len(keys) / max(1, len(h))))
    return keys_to_text(keys), h, stats


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from flyhamlet.entropy_tap import read_isi_file
    ap = argparse.ArgumentParser()
    ap.add_argument("isi_files", nargs="+")
    ap.add_argument("--out", default=None, help="write the key text here (default: stdout)")
    ap.add_argument("--hash-bin", default=None, help="also write the raw SHA-256 byte stream")
    ap.add_argument("--batch", type=int, default=BATCH)
    a = ap.parse_args(argv)
    isis = np.concatenate([read_isi_file(f)[1] for f in a.isi_files])
    text, h, stats = whiten(isis, a.batch)
    print(f"whiten: {stats}", file=sys.stderr)
    if a.hash_bin:
        Path(a.hash_bin).write_bytes(h)
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
