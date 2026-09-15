"""Opt-in, spike-driven recurrent prosthesis for FlyHamlet's Python simulator.

The fly's synapses remain fixed. Only ``fit_feedback`` changes the prosthesis's
write-back weights; its input/recurrent matrices form a fixed reservoir.
``observe`` receives spikes only, never a cue label, reward, or trial number.
All units here are artificial continuous-valued units, not reconstructed cells.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProsthesisConfig:
    n_units: int = 256
    update_ms: float = 10.0
    tau_ms: float = 500.0
    recurrent_gain: float = 0.98
    input_gain: float = 1.0
    rate_scale_hz: float = 100.0
    max_drive_mV: float = 3.0
    seed: int = 6
    mode: str = "recurrent"  # recurrent | leaky | memoryless

    def __post_init__(self):
        if isinstance(self.n_units, bool) or not isinstance(self.n_units, (int, np.integer)) or self.n_units < 1:
            raise ValueError("n_units must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("update_ms", "tau_ms", "input_gain", "rate_scale_hz", "max_drive_mV"):
            x = getattr(self, name)
            if not np.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.recurrent_gain) or not 0 <= self.recurrent_gain < 1:
            raise ValueError("recurrent_gain must lie in [0, 1)")
        if self.mode not in {"recurrent", "leaky", "memoryless"}:
            raise ValueError("mode must be recurrent, leaky, or memoryless")


def _validate_groups(groups: Sequence[Sequence[int]], n: int, name: str) -> tuple[np.ndarray, ...]:
    if len(groups) == 0:
        raise ValueError(f"{name} needs at least one group")
    result = []
    for group in groups:
        a = np.asarray(group)
        if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "iu":
            raise ValueError(f"{name} groups must be nonempty one-dimensional integer arrays")
        if np.any(a < 0) or np.any(a >= n):
            raise ValueError(f"{name} neuron index outside [0, {n})")
        a = a.astype(np.int64, copy=True)
        if np.unique(a).size != a.size:
            raise ValueError(f"duplicate neuron within {name} group")
        a.setflags(write=False)
        result.append(a)
    flat = np.concatenate(result)
    if np.unique(flat).size != flat.size:
        raise ValueError(f"overlapping {name} groups would double-count neurons")
    return tuple(result)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Fit y = x @ weights.T + bias; training data only, unpenalized intercept.

    Uses a dual solve for fewer samples than features. No sklearn dependency.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("x and y must be 2-D with the same >=2 samples")
    if x.shape[1] == 0 or y.shape[1] == 0 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonempty, finite feature/target matrices required")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    if x.shape[0] < x.shape[1]:
        gram = xc @ xc.T + ridge * np.eye(x.shape[0])
        w = (xc.T @ np.linalg.solve(gram, yc)).T
    else:
        gram = xc.T @ xc + ridge * np.eye(x.shape[1])
        w = np.linalg.solve(gram, xc.T @ yc).T
    b = ym - w @ xm
    if not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise FloatingPointError("ridge fit produced nonfinite weights")
    return w, b


class RecurrentProsthesis:
    """Pooled spikes -> bounded recurrent state -> bounded synaptic drive.

    read_groups/write_groups contain *dense indices*, not FlyWire root IDs.
    Resolve annotations/root IDs with ``net.resolve(spec)`` first. Groups within
    each port must be disjoint. Read and write ports may overlap deliberately.

    A completed spike bin updates the lobe after the fly timestep finishes.
    The new drive first enters the NEXT timestep's synaptic slot. The simulator
    adds (1 - exp(-dt/tau_syn)) * drive to g, discarding input to refractory
    neurons and resetting spiking cells normally. Thus max_drive_mV bounds the
    prosthetic steady-state g contribution in a nonspiking cell; it is neither
    a physical conductance nor a bound on total network activity.
    """

    def __init__(self, n_neurons: int, dt_ms: float,
                 read_groups: Sequence[Sequence[int]], write_groups: Sequence[Sequence[int]],
                 config: ProsthesisConfig | None = None):
        self.config = config or ProsthesisConfig()
        if isinstance(n_neurons, bool) or not isinstance(n_neurons, (int, np.integer)) or n_neurons < 1:
            raise ValueError("n_neurons must be a positive integer")
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be positive and finite")
        self.n_neurons, self.dt_ms = int(n_neurons), float(dt_ms)
        bins = self.config.update_ms / self.dt_ms
        if bins < 1 or not math.isclose(bins, round(bins), rel_tol=0, abs_tol=1e-8):
            raise ValueError("update_ms must be an integer multiple of dt_ms")
        self.bin_steps = int(round(bins))
        self.read_groups = _validate_groups(read_groups, self.n_neurons, "read")
        self.write_groups = _validate_groups(write_groups, self.n_neurons, "write")
        self.read_sizes = np.array([len(g) for g in self.read_groups], dtype=float)
        self.write_indices = np.concatenate(self.write_groups)
        self.write_channel = np.repeat(np.arange(len(self.write_groups)), [len(g) for g in self.write_groups])
        self.write_indices.setflags(write=False)
        self.write_channel.setflags(write=False)
        self._channel = np.full(self.n_neurons, -1, dtype=np.int32)
        for i, g in enumerate(self.read_groups):
            self._channel[g] = i
        self._alpha = -math.expm1(-self.config.update_ms / self.config.tau_ms)
        rng = np.random.default_rng(self.config.seed)  # never consumes the fly's RNG
        units = self.config.n_units
        self.w_in = rng.normal(0, self.config.input_gain / np.sqrt(len(self.read_groups)),
                               (units, len(self.read_groups)))
        # Q has spectral norm 1; gain < 1 keeps the isolated reservoir contractive.
        q, _ = np.linalg.qr(rng.normal(size=(units, units)))
        self.w_rec = self.config.recurrent_gain * q
        if self.config.mode != "recurrent":
            self.w_rec.fill(0)
        self.w_out = np.zeros((len(self.write_groups), units))
        self.bias = np.zeros(len(self.write_groups))
        self.feedback_enabled = True
        self.observation_enabled = True
        self._owner = None
        self.reset_state()

    def reset_state(self):
        """Erase all trial history, including partial bins; retain learned weights and gates."""
        self.state = np.zeros(self.config.n_units)
        self.last_rates_hz = np.zeros(len(self.read_groups))
        self.drive_mV = np.zeros(len(self.write_groups))
        self._counts = np.zeros(len(self.read_groups), dtype=np.int64)
        self._ticks = 0
        self.n_updates = 0
        self.saturated_outputs = 0
        self.peak_abs_drive_mV = 0.0

    def observe(self, spikes: np.ndarray) -> bool:
        """Consume one timestep's spikes. Return True at a completed sampling bin."""
        # The simulator guarantees valid, unique indices. Public callers get validation.
        spikes = np.asarray(spikes)
        if spikes.ndim != 1 or spikes.dtype.kind not in "iu":
            raise ValueError("spikes must be a one-dimensional integer array")
        if spikes.size and (np.any(spikes < 0) or np.any(spikes >= self.n_neurons)):
            raise ValueError("spike index out of range")
        if self.observation_enabled and spikes.size:
            channels = self._channel[spikes]
            channels = channels[channels >= 0]
            if channels.size:
                self._counts += np.bincount(channels, minlength=len(self.read_groups))
        self._ticks += 1
        if self._ticks < self.bin_steps:
            return False
        self.last_rates_hz = self._counts / (self.read_sizes * self.config.update_ms * 1e-3)
        features = np.clip(self.last_rates_hz / self.config.rate_scale_hz, 0, 1)
        z = self.w_in @ features
        if self.config.mode == "recurrent":
            z += self.w_rec @ self.state
        candidate = np.tanh(z)
        if self.config.mode == "memoryless":
            self.state = candidate
        else:
            self.state = (1 - self._alpha) * self.state + self._alpha * candidate
        raw = self.w_out @ self.state + self.bias
        if not (np.isfinite(self.state).all() and np.isfinite(raw).all()):
            self.drive_mV.fill(0)
            raise FloatingPointError("nonfinite prosthesis state/output; feedback stopped")
        cap = self.config.max_drive_mV
        self.saturated_outputs += int(np.count_nonzero(np.abs(raw) > cap))
        self.drive_mV = np.clip(raw, -cap, cap)
        self.peak_abs_drive_mV = max(self.peak_abs_drive_mV, float(np.abs(self.drive_mV).max()))
        self._counts.fill(0)
        self._ticks = 0
        self.n_updates += 1
        return True

    def fit_feedback(self, states: np.ndarray, target_drive_mV: np.ndarray,
                     ridge: float = 1e-3) -> float:
        """Supervised calibration of write-back weights only; return training MSE.

        Call outside evaluation. Labels can construct training targets here but
        must never enter ``observe``. After fitting, reset trial state before use.
        """
        states = np.asarray(states, dtype=float)
        targets = np.asarray(target_drive_mV, dtype=float)
        if states.ndim != 2 or states.shape[1] != self.config.n_units:
            raise ValueError("states must have n_units columns")
        if targets.ndim != 2 or targets.shape[1] != len(self.write_groups):
            raise ValueError("targets must have one column per write group")
        if np.any(np.abs(targets) > self.config.max_drive_mV):
            raise ValueError("training target exceeds max_drive_mV")
        w, b = fit_ridge(states, targets, ridge)
        self.w_out, self.bias = w, b
        self.reset_state()
        return float(np.mean((states @ w.T + b - targets) ** 2))

    def weights_digest(self) -> str:
        h = hashlib.sha256()
        for w in (self.w_in, self.w_rec, self.w_out, self.bias):
            h.update(np.ascontiguousarray(w, dtype="<f8").tobytes())
        return h.hexdigest()

    def _metadata(self) -> dict:
        return {"format": 1, "config": asdict(self.config), "n_neurons": self.n_neurons,
                "dt_ms": self.dt_ms, "read_groups": [g.tolist() for g in self.read_groups],
                "write_groups": [g.tolist() for g in self.write_groups]}

    def save_weights(self, path: str | Path):
        """Save parameters and exact ports, excluding transient trial state."""
        with open(path, "wb") as f:
            np.savez_compressed(f, metadata=json.dumps(self._metadata(), sort_keys=True),
                                w_in=self.w_in, w_rec=self.w_rec, w_out=self.w_out, bias=self.bias)

    def load_weights(self, path: str | Path):
        """Load into a matching configuration; reject incompatible/nonfinite parameters."""
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["metadata"])) != self._metadata():
                raise ValueError("checkpoint ports/configuration differ from this prosthesis")
            arrays = {name: np.array(z[name], dtype=float, copy=True)
                      for name in ("w_in", "w_rec", "w_out", "bias")}
        for name, value in arrays.items():
            if value.shape != getattr(self, name).shape or not np.isfinite(value).all():
                raise ValueError(f"invalid checkpoint array {name}")
        for name, value in arrays.items():
            setattr(self, name, value)
        self.reset_state()
