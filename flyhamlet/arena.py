"""Phase 2: a simulated fly walking in a 2D arena, driven by its connectome.

* Motor readout: descending-neuron spike counts smoothed over a short window.
  Angular velocity comes from the left/right asymmetry of the turning DNs
  (DNa02 + DNa01), forward speed from DNp09 minus the backward-walking MDNs,
  plus a configurable base speed (the network is silent without input).
* Sensory input: when a wall comes within ``wall_distance_mm`` along an eye's
  rays, the looming-detecting optic glomeruli (LC4, LPLC2) of that eye receive
  Poisson input scaled by proximity. Everything goes through the generic
  ``LIFNetwork`` input API, so other senses can be added the same way.
* The arena is also a typewriter (see :mod:`typewriter`).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .data import Connectome
from .sim import LIFNetwork
from .typewriter import Typewriter


# --------------------------------------------------------------------------- geometry
def ray_to_wall(x: float, y: float, phi: float, w: float, h: float) -> float:
    """Distance from (x, y) along direction ``phi`` (rad) to the rectangle [0,w]x[0,h]."""
    c, s = math.cos(phi), math.sin(phi)
    d = math.inf
    if c > 1e-9:
        d = min(d, (w - x) / c)
    elif c < -1e-9:
        d = min(d, -x / c)
    if s > 1e-9:
        d = min(d, (h - y) / s)
    elif s < -1e-9:
        d = min(d, -y / s)
    return max(0.0, d)


class WindowedRate:
    """Spike rate of a neuron set over a sliding window of ``n_steps`` steps."""

    def __init__(self, n_steps: int, dt_ms: float, n_neurons: int):
        self.buf = np.zeros(n_steps, dtype=np.int32)
        self.i = 0
        self.total = 0
        self.scale = 1.0 / (n_steps * dt_ms * 1e-3 * max(1, n_neurons))   # -> Hz per neuron

    def push(self, count: int):
        self.total += count - self.buf[self.i]
        self.buf[self.i] = count
        self.i = (self.i + 1) % len(self.buf)

    @property
    def hz(self) -> float:
        return self.total * self.scale


@dataclass
class FlyState:
    x: float
    y: float
    heading: float        # radians, CCW from +x
    v: float = 0.0        # mm/s (negative = backward)
    omega: float = 0.0    # rad/s


class FlyArena:
    """One fly, one connectome, one arena. Call :meth:`run` or :meth:`step_control`."""

    def __init__(self, c: Connectome, cfg: dict, seed: int, backend: str = "numba", taps: list | None = None):
        self.cfg = cfg
        acfg = cfg["arena"]; mcfg = acfg["motor"]; lcfg = acfg["looming"]
        self.rng = np.random.default_rng(seed)
        self.net = LIFNetwork(c, cfg["sim"], seed=seed, backend=backend)
        self.dt_ms = self.net.dt
        self.W, self.H = float(acfg["width_mm"]), float(acfg["height_mm"])
        self.control_steps = max(1, int(round(float(acfg["control_interval_ms"]) / self.dt_ms)))
        self.log_steps = max(1, int(round(float(acfg["log_interval_ms"]) / self.dt_ms)))

        # ----- motor readout
        win = int(round(float(mcfg["window_ms"]) / self.dt_ms))
        side = c.neurons["side"].to_numpy()
        self.groups: dict[str, tuple] = {}
        def add_group(name, types, s):
            idx = np.concatenate([self.net.resolve(t) for t in types])
            idx = idx[side[idx] == s]
            if len(idx) == 0:
                raise KeyError(f"no {s} neurons for types {types}")
            sub = self.net.subscribe({"index": idx}, name)
            self.groups[name] = (sub, WindowedRate(win, self.dt_ms, len(idx)), idx)
        add_group("turn_L", mcfg["turn_types"], "left"); add_group("turn_R", mcfg["turn_types"], "right")
        for s in ("left", "right"):
            add_group(f"fwd_{s[0].upper()}", mcfg["forward_types"], s)
            add_group(f"bwd_{s[0].upper()}", mcfg["backward_types"], s)
        self.gf = self.net.subscribe("Giant_Fiber", "GF")
        self.gf_rate = WindowedRate(win, self.dt_ms, len(self.gf.indices))
        self.m = dict(base=float(mcfg["base_speed_mm_s"]), sg=float(mcfg["speed_gain"]), bg=float(mcfg["backward_gain"]),
                      tg=math.radians(float(mcfg["turn_gain"])), vmax=float(mcfg["max_speed_mm_s"]),
                      wmax=math.radians(float(mcfg["max_turn_deg_s"])), sign=float(mcfg["turn_sign"]))

        # ----- looming input (one Poisson handle per eye; rate set every control step)
        self.loom = dict(d=float(lcfg["wall_distance_mm"]), rmax=float(lcfg["max_rate_hz"]), k=float(lcfg["exponent"]))
        self.eye_rays = [math.radians(a) for a in lcfg["eye_ray_angles_deg"]]
        self.eye_h = {"left": self.net.inject_poisson({"type": list(lcfg["types"]), "side": "left"}, 0.0),
                      "right": self.net.inject_poisson({"type": list(lcfg["types"]), "side": "right"}, 0.0)}
        self.eye_rate = {"left": 0.0, "right": 0.0}

        # ----- fly, typewriter, logs
        tw = acfg["typewriter"]
        self.tw = Typewriter(self.W, self.H, int(tw["rows"]), int(tw["cols"]), tw["alphabet"], int(tw["layout_seed"]))
        if acfg.get("start", "random") == "center":
            x, y, h = self.W / 2, self.H / 2, 0.0
        else:
            x, y = self.rng.uniform(0.1 * self.W, 0.9 * self.W), self.rng.uniform(0.1 * self.H, 0.9 * self.H)
            h = self.rng.uniform(-math.pi, math.pi)
        self.fly = FlyState(x, y, h)
        self.tw.start(x, y)
        self.traj: list[tuple] = []
        self.taps = list(taps or [])          # objects with .on_step(step, spikes) hooks (EntropyTap)
        for tap in self.taps:
            tap.attach(self.net)
        self.n_spikes = 0
        self.wall_steps = 0

    # ----------------------------------------------------------------- sensing / acting
    def _sense(self):
        f, L = self.fly, self.loom
        for eye, sgn in (("left", 1.0), ("right", -1.0)):
            d = min(ray_to_wall(f.x, f.y, f.heading + sgn * a, self.W, self.H) for a in self.eye_rays)
            prox = max(0.0, 1.0 - d / L["d"])
            rate = L["rmax"] * prox ** L["k"]
            self.eye_rate[eye] = rate
            self.net.set_poisson_rate(self.eye_h[eye], rate)

    def _act(self, dt_s: float):
        g, m, f = self.groups, self.m, self.fly
        turn = g["turn_L"][1].hz - g["turn_R"][1].hz
        fwd = 0.5 * (g["fwd_L"][1].hz + g["fwd_R"][1].hz)
        bwd = 0.5 * (g["bwd_L"][1].hz + g["bwd_R"][1].hz)
        f.omega = max(-m["wmax"], min(m["wmax"], m["sign"] * m["tg"] * turn))
        f.v = max(-m["vmax"], min(m["vmax"], m["base"] + m["sg"] * fwd - m["bg"] * bwd))
        f.heading = (f.heading + f.omega * dt_s + math.pi) % (2 * math.pi) - math.pi
        nx = f.x + f.v * math.cos(f.heading) * dt_s
        ny = f.y + f.v * math.sin(f.heading) * dt_s
        if not (0.0 <= nx <= self.W and 0.0 <= ny <= self.H):
            self.wall_steps += 1
        f.x = min(self.W, max(0.0, nx)); f.y = min(self.H, max(0.0, ny))

    def step_control(self) -> str | None:
        """Advance one control interval (``control_steps`` network steps). Returns a typed letter or None."""
        for _ in range(self.control_steps):
            spikes = self.net.step()
            self.n_spikes += len(spikes)
            for name, (sub, wr, _) in self.groups.items():
                wr.push(sub.count)
            self.gf_rate.push(self.gf.count)
            for tap in self.taps:
                tap.on_step(self.net.step_idx - 1, spikes)
        self._act(self.control_steps * self.dt_ms * 1e-3)
        self._sense()
        f = self.fly
        t = self.net.time_s
        if (self.net.step_idx // self.control_steps) % max(1, self.log_steps // self.control_steps) == 0:
            self.traj.append((t, f.x, f.y, math.degrees(f.heading), f.v, math.degrees(f.omega),
                              self.eye_rate["left"], self.eye_rate["right"],
                              self.groups["turn_L"][1].hz, self.groups["turn_R"][1].hz,
                              0.5 * (self.groups["fwd_L"][1].hz + self.groups["fwd_R"][1].hz),
                              0.5 * (self.groups["bwd_L"][1].hz + self.groups["bwd_R"][1].hz), self.gf_rate.hz))
        return self.tw.update(t, f.x, f.y, math.degrees(f.heading))

    def run(self, duration_s: float, on_control: Callable | None = None):
        n = int(round(duration_s * 1e3 / (self.control_steps * self.dt_ms)))
        for i in range(n):
            letter = self.step_control()
            if on_control is not None:
                on_control(self, i, letter)
        return self

    # ----------------------------------------------------------------- output
    TRAJ_COLS = ["t_s", "x", "y", "heading_deg", "v_mm_s", "omega_deg_s", "loom_L_hz", "loom_R_hz",
                 "turnDN_L_hz", "turnDN_R_hz", "fwdDN_hz", "bwdDN_hz", "GF_hz"]

    def save(self, out_dir: str | Path, fly_id: int, seed: int):
        import pandas as pd
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.tw.write_csv(out / f"fly{fly_id:02d}_keystrokes.csv")
        pd.DataFrame(self.traj, columns=self.TRAJ_COLS).to_csv(out / f"fly{fly_id:02d}_trajectory.csv", index=False, float_format="%.3f")
        summ = dict(fly=fly_id, seed=seed, sim_time_s=self.net.time_s, keystrokes=len(self.tw.log),
                    keystrokes_per_s=len(self.tw.log) / max(1e-9, self.net.time_s), spikes=self.n_spikes,
                    wall_contact_fraction=self.wall_steps / max(1, self.net.step_idx // self.control_steps),
                    text=self.tw.text(), layout=self.tw.letters)
        json.dump(summ, open(out / f"fly{fly_id:02d}_summary.json", "w"), indent=1)
        for tap in self.taps:
            tap.close()
        return summ
