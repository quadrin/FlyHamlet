"""Optional matplotlib live view of one fly in the typewriter arena."""
from __future__ import annotations

import math

import numpy as np


class LiveView:
    def __init__(self, arena, every: int = 100, trail: int = 3000):
        import matplotlib.pyplot as plt
        self.plt = plt; self.a = arena; self.every = every
        self.xs, self.ys = [], []
        self.trail = trail
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        tw = arena.tw
        for r in range(tw.rows):
            for c in range(tw.cols):
                self.ax.add_patch(plt.Rectangle((c * tw.width / tw.cols, r * tw.height / tw.rows), tw.width / tw.cols,
                                                tw.height / tw.rows, fill=False, lw=0.5, color="0.6"))
                self.ax.text((c + 0.5) * tw.width / tw.cols, (r + 0.5) * tw.height / tw.rows,
                             repr(tw.letters[r * tw.cols + c]), ha="center", va="center", color="0.5", fontsize=8)
        self.ax.set_xlim(0, tw.width); self.ax.set_ylim(0, tw.height); self.ax.set_aspect("equal")
        (self.line,) = self.ax.plot([], [], "-", lw=0.7, color="tab:blue")
        (self.dot,) = self.ax.plot([], [], "o", color="tab:red")
        self.arrow = None
        self.title = self.ax.set_title("")
        self.fig.tight_layout()

    def __call__(self, arena, i, letter):
        f = arena.fly
        self.xs.append(f.x); self.ys.append(f.y)
        if i % self.every:
            return
        xs, ys = self.xs[-self.trail:], self.ys[-self.trail:]
        self.line.set_data(xs, ys); self.dot.set_data([f.x], [f.y])
        if self.arrow is not None:
            self.arrow.remove()
        self.arrow = self.ax.arrow(f.x, f.y, 3 * math.cos(f.heading), 3 * math.sin(f.heading), head_width=1.0, color="tab:red")
        g = arena.groups
        self.title.set_text(f"t={arena.net.time_s:6.2f}s  v={f.v:5.1f} mm/s  w={math.degrees(f.omega):6.0f} deg/s  "
                            f"loom L/R={arena.eye_rate['left']:.0f}/{arena.eye_rate['right']:.0f} Hz  "
                            f"turnDN L/R={g['turn_L'][1].hz:.0f}/{g['turn_R'][1].hz:.0f}  typed: {arena.tw.text()[-30:]!r}")
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events(); self.plt.pause(0.001)
