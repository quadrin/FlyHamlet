"""Read-only activity guard for experimental runs; thresholds are engineering limits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .sim import LIFNetwork


@dataclass(frozen=True)
class ActivityLimits:
    window_ms: float = 50.0
    max_mean_hz: float = 100.0
    max_cell_hz: float = 500.0
    max_group_hz: float = 250.0

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("activity limits must be positive and finite")


class ActivityAbort(RuntimeError):
    def __init__(self, details: dict):
        self.details = details
        super().__init__(details["reason"])


class ActivityMonitor:
    """Count every spike, checking fixed nonoverlapping windows and final partial bins.

    Checks finite v/g at the same boundaries. It neither consumes random numbers
    nor writes neural state. A tripped guard stays latched until a trial reset.
    """
    def __init__(self, n: int, dt_ms: float, limits: ActivityLimits, groups=None):
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be positive and finite")
        ratio = limits.window_ms / dt_ms
        if n < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
            raise ValueError("window_ms must be an integer multiple of dt_ms")
        self.n, self.dt_ms, self.limits = n, dt_ms, limits
        self.window_steps = int(round(ratio))
        self.set_groups(groups or {})
        self.reset()

    def set_groups(self, groups):
        self.groups = {}
        for name, group in groups.items():
            a = np.asarray(group)
            if (a.ndim != 1 or not len(a) or a.dtype.kind not in "iu" or
                    np.any(a < 0) or np.any(a >= self.n) or len(np.unique(a)) != len(a)):
                raise ValueError(f"invalid monitored group: {name}")
            self.groups[name] = a.astype(np.int64)

    def reset(self):
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.bin_steps = self.total_steps = 0
        self.windows = []
        self.aborted = None

    def observe(self, net, spikes):
        if self.aborted is not None:
            raise ActivityAbort(self.aborted)
        np.add.at(self.counts, spikes, 1)
        self.bin_steps += 1
        self.total_steps += 1
        if self.bin_steps == self.window_steps:
            self.flush(net)

    def flush(self, net):
        if self.aborted is not None:
            raise ActivityAbort(self.aborted)
        if not self.bin_steps:
            return
        duration = self.bin_steps * self.dt_ms * 1e-3
        rates = self.counts / duration
        window = dict(end_step=self.total_steps, duration_ms=duration * 1e3,
                      mean_hz=float(rates.mean()), max_cell_hz=float(rates.max()),
                      group_hz={k: float(rates[v].mean()) for k, v in self.groups.items()})
        if net.backend == "torch":
            finite = bool(net.t_v.isfinite().all().item() and net.t_g.isfinite().all().item())
        else:
            finite = bool(np.isfinite(net.v).all() and np.isfinite(net.g).all())
        self.windows.append(window)
        reason = None
        if not finite:
            reason = "nonfinite neural state"
        elif window["mean_hz"] > self.limits.max_mean_hz:
            reason = "whole-network mean firing-rate limit exceeded"
        elif window["max_cell_hz"] > self.limits.max_cell_hz:
            reason = "single-neuron firing-rate limit exceeded"
        elif any(v > self.limits.max_group_hz for v in window["group_hz"].values()):
            reason = "monitored-population firing-rate limit exceeded"
        if reason:
            self.aborted = dict(reason=reason, noise_seed=int(net.seed),
                                limits=asdict(self.limits), window=window)
            raise ActivityAbort(self.aborted)
        self.counts.fill(0)
        self.bin_steps = 0


class GuardedNetwork(LIFNetwork):
    """Opt-in wrapper used by screening/benchmarks, leaving the base simulator intact."""
    def __init__(self, *args, limits=None, groups=None, activity_records=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.activity = ActivityMonitor(self.n, self.dt, limits or ActivityLimits(), groups)
        self.activity_records = activity_records if activity_records is not None else []
        self._trial_finished = False

    def step(self):
        if self._trial_finished:
            raise RuntimeError("reset_state before continuing a finished trial")
        if self.activity.aborted is not None:
            raise ActivityAbort(self.activity.aborted)
        spikes = super().step()
        self.activity.observe(self, spikes)
        return spikes

    def finish_trial(self):
        if self._trial_finished or not self.activity.total_steps:
            return
        self.activity.flush(self)
        self.activity_records.append(dict(noise_seed=int(self.seed),
                                          n_steps=self.activity.total_steps,
                                          windows=list(self.activity.windows)))
        self._trial_finished = True

    def reset_state(self, seed=None):
        if self.activity.aborted is None:
            self.finish_trial()
        else:
            self.activity_records.append(dict(noise_seed=int(self.seed),
                    n_steps=self.activity.total_steps, windows=list(self.activity.windows),
                    aborted=self.activity.aborted))
        super().reset_state(seed=seed)
        self.activity.reset()
        self._trial_finished = False
