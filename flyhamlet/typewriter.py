"""A 2D arena as a typewriter: a rows x cols grid of key regions, one letter each."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Typewriter:
    width: float
    height: float
    rows: int = 3
    cols: int = 9
    alphabet: str = "abcdefghijklmnopqrstuvwxyz "
    layout_seed: int = 7
    letters: list[str] = field(init=False)
    log: list[tuple] = field(default_factory=list)
    _last_key: int = -1

    def __post_init__(self):
        assert self.rows * self.cols == len(self.alphabet), "grid must have one key per symbol"
        perm = np.random.default_rng(self.layout_seed).permutation(len(self.alphabet))
        self.letters = [self.alphabet[p] for p in perm]

    def key_at(self, x: float, y: float) -> int:
        col = min(self.cols - 1, max(0, int(x / self.width * self.cols)))
        row = min(self.rows - 1, max(0, int(y / self.height * self.rows)))
        return row * self.cols + col

    def layout_str(self) -> str:
        return "\n".join(" ".join(repr(self.letters[r * self.cols + c]) for c in range(self.cols)) for r in range(self.rows))

    def start(self, x: float, y: float):
        """Mark the starting region so it is not logged as an entry."""
        self._last_key = self.key_at(x, y)

    def update(self, t_s: float, x: float, y: float, heading_deg: float) -> str | None:
        """Log a keystroke if (x, y) entered a new key region. Returns the letter or None."""
        k = self.key_at(x, y)
        if k == self._last_key:
            return None
        self._last_key = k
        self.log.append((t_s, k, self.letters[k], x, y, heading_deg))
        return self.letters[k]

    def text(self) -> str:
        return "".join(r[2] for r in self.log)

    def write_csv(self, path: str | Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_time_s", "key_index", "letter", "x", "y", "heading_deg"])
            for t, k, l, x, y, h in self.log:
                w.writerow([f"{t:.4f}", k, l, f"{x:.3f}", f"{y:.3f}", f"{h:.2f}"])
