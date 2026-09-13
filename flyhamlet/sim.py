"""Phase 1: sparse leaky integrate-and-fire simulation of the whole FlyWire connectome.

Model (Shiu et al. 2024, Nature; Brian2 reference implementation)::

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike when v > v_th; reset v = v_rst, g = 0; refractory t_rfc
    presynaptic spike -> g_post += w after delay t_dly, DROPPED if post is refractory
    w = sign(pre) * syn_count * w_syn        (ACh/DA/OCT/SER +, GABA/GLUT -)

The two linear ODEs are integrated exactly per timestep (Brian2 ``method='linear'``).
Update order within a step mirrors Brian2's default schedule
(state update -> threshold -> synapses/Poisson -> reset).  As in Brian2, a
variable flagged ``unless refractory`` ignores synaptic updates while the neuron
is refractory: input arriving during the 2.2 ms after a spike is discarded
(verified step-by-step against Brian2 2.9, see tests/test_sim.py).

Determinism
-----------
The network state is fully determined by the connectome and the parameters.
Without any input and without background noise every neuron sits at rest
(v = v_0, g = 0) and no spike ever occurs: the simulation is trivially
deterministic. All stochastic inputs (Poisson injections, background drive)
draw from a single ``numpy.random.Generator`` seeded by ``seed``; two runs with
the same seed, the same inputs registered in the same order and the same
backend produce identical spike trains. The numba and torch backends agree up
to float32 summation order.

Backends
--------
``numba`` (default, CPU): fused loops over the 139k neurons; spike delivery
walks CSR rows of the spiking neurons only.
``torch``: same algorithm in PyTorch ops; on a CUDA device the delivery is a
gather + ``index_add_``.  Use ``backend="auto"`` to pick torch when a GPU is
visible and numba otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .data import Connectome

try:  # optional
    import numba as _nb
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _nb = None
    HAVE_NUMBA = False

try:  # optional
    import torch as _torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    _torch = None
    HAVE_TORCH = False


# --------------------------------------------------------------------------- targets
def resolve_target(c: Connectome, target, side: str | None = None) -> np.ndarray:
    """Resolve a neuron target specification to a sorted array of dense indices.

    ``target`` may be

    * a ``str``: a cell type name, matched exactly against the FlyWire ``cell_type``,
      ``hemibrain_type`` and consolidated type columns (case sensitive);
    * an iterable of ``int``: FlyWire root ids (>= 2**40) or dense indices;
    * a ``dict`` with one of ``ids``, ``index``, ``type`` (+ optional ``side``),
      ``label`` (regex over community labels), ``super_class`` (list), ``mask`` (bool array);
    * a boolean mask of length ``c.n``.
    """
    if isinstance(target, str):
        idx = c.neurons_of_type(target, side=side)
        if len(idx) == 0:
            raise KeyError(f"cell type {target!r} not found in v783 annotations "
                           f"(try Connectome.search_types)")
        return idx
    if isinstance(target, dict):
        if "ids" in target:
            idx = c.index_of(target["ids"])
        elif "index" in target:
            idx = np.asarray(target["index"], dtype=np.int64)
        elif "type" in target:
            names = target["type"] if isinstance(target["type"], (list, tuple)) else [target["type"]]
            idx = np.concatenate([resolve_target(c, t, side=target.get("side", side)) for t in names])
        elif "label" in target:
            idx = c.neurons_with_label(target["label"])
        elif "super_class" in target:
            idx = np.flatnonzero(c.super_class_mask(target["super_class"]))
        elif "mask" in target:
            idx = np.flatnonzero(np.asarray(target["mask"], dtype=bool))
        else:
            raise ValueError(f"unknown target spec {target}")
        if target.get("side") is not None and "type" not in target:
            idx = idx[(c.neurons["side"].to_numpy()[idx] == target["side"])]
        return np.unique(idx)
    arr = np.asarray(target)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    arr = arr.astype(np.int64).ravel()
    if len(arr) and arr.min() >= (1 << 40):
        return np.unique(c.index_of(arr))
    return np.unique(arr)


# --------------------------------------------------------------------------- numba kernels
if HAVE_NUMBA:
    @_nb.njit(cache=True, fastmath=False)
    def _nb_integrate(v, g, i_ext, rfc_left, frozen, v0, a, b, cc, v_th, spikes_out):
        """Exact LIF integration for one step + threshold. Returns number of spikes.
        ``frozen[i]`` is set when neuron i was refractory in this step (its inputs are dropped)."""
        n = v.shape[0]
        k = 0
        for i in range(n):
            if rfc_left[i] > 0:
                rfc_left[i] -= 1
                frozen[i] = True
                continue
            frozen[i] = False
            gi = g[i]
            ss = v0 + i_ext[i]
            vi = ss + a * (v[i] - ss) + b * gi
            v[i] = vi
            g[i] = cc * gi
            if vi > v_th:
                spikes_out[k] = i
                k += 1
        return k

    @_nb.njit(cache=True)
    def _nb_deliver(indptr, indices, w, spikes, g, frozen):
        for s in spikes:
            for p in range(indptr[s], indptr[s + 1]):
                j = indices[p]
                if not frozen[j]:
                    g[j] += w[p]

    @_nb.njit(cache=True)
    def _nb_add_at(x, idx, val, frozen):
        for i in range(idx.shape[0]):
            if not frozen[idx[i]]:
                x[idx[i]] += val

    @_nb.njit(cache=True)
    def _nb_add_at_w(x, idx, vals, frozen):
        for i in range(idx.shape[0]):
            if not frozen[idx[i]]:
                x[idx[i]] += vals[i]

    @_nb.njit(cache=True)
    def _nb_reset(v, g, rfc_left, rfc_steps, spikes, v_rst):
        for s in spikes:
            v[s] = v_rst
            g[s] = 0.0
            rfc_left[s] = rfc_steps[s]


# --------------------------------------------------------------------------- records
@dataclass
class Subscription:
    """Spike events from a neuron set, delivered per timestep.

    After every step ``last`` holds the member indices that spiked in that step and
    ``count`` their number; ``callback(step, spikes)`` is invoked if given.
    """
    name: str
    indices: np.ndarray
    member: np.ndarray            # bool mask over all neurons
    callback: Callable | None = None
    last: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    count: int = 0
    total: int = 0

    def deliver(self, step: int, spikes: np.ndarray):
        self.last = spikes[self.member[spikes]] if len(spikes) else spikes
        self.count = len(self.last)
        self.total += self.count
        if self.callback is not None:
            self.callback(step, self.last)


@dataclass
class SpikeRecord:
    """All spikes of a run as parallel arrays (step, neuron index)."""
    dt_ms: float
    steps: np.ndarray
    neurons: np.ndarray
    n_steps: int

    @property
    def times_s(self) -> np.ndarray:
        return self.steps * self.dt_ms * 1e-3

    def rates_hz(self, n_neurons: int) -> np.ndarray:
        dur = self.n_steps * self.dt_ms * 1e-3
        return np.bincount(self.neurons, minlength=n_neurons) / dur


@dataclass
class _PoissonInput:
    indices: np.ndarray
    p: float                   # per-step event probability = rate * dt
    weight: float
    to_v: bool


# --------------------------------------------------------------------------- network
class LIFNetwork:
    """Sparse LIF network over the full connectome.

    Parameters
    ----------
    c : Connectome
    sim_cfg : dict           ``config.yaml['sim']``
    seed : int | None        overrides ``sim_cfg['seed']``
    backend : 'auto' | 'numba' | 'numpy' | 'torch'
    device : torch device string (torch backend only)
    """

    def __init__(self, c: Connectome, sim_cfg: dict, seed: int | None = None,
                 backend: str = "auto", device: str | None = None):
        self.c = c
        self.cfg = dict(sim_cfg)
        self.n = c.n
        self.dt = float(sim_cfg["dt_ms"])
        self.seed = int(sim_cfg["seed"] if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        dtype = np.float32 if str(sim_cfg.get("dtype", "float32")) == "float32" else np.float64
        self.dtype = dtype

        # constants (mV, ms)
        self.v0 = float(sim_cfg["v_rest_mV"]); self.v_rst = float(sim_cfg["v_reset_mV"])
        self.v_th = float(sim_cfg["v_thresh_mV"])
        tm, ts = float(sim_cfg["tau_mem_ms"]), float(sim_cfg["tau_syn_ms"])
        self.a = math.exp(-self.dt / tm)              # membrane decay per step
        self.cc = math.exp(-self.dt / ts)             # synaptic decay per step
        self.b = ts / (ts - tm) * (self.cc - self.a)  # exact coupling term
        # Brian2 semantics: a neuron resumes integrating at the first step with
        # t - t_spike >= t_rfc, i.e. it is frozen for round(t_rfc/dt) - 1 steps (21 for 2.2 ms / 0.1 ms).
        self.rfc_default = max(0, int(round(float(sim_cfg["t_refractory_ms"]) / self.dt)) - 1)
        self.delay_steps = max(1, int(round(float(sim_cfg["delay_ms"]) / self.dt)))
        self.w_syn = float(sim_cfg["w_syn_mV"])
        self.poisson_weight = float(sim_cfg.get("poisson_weight_mV", 68.75))
        # store the per-step constants in the state dtype so every backend evaluates the
        # same float32 expression and produces bit-identical spike trains
        f = np.dtype(dtype).type
        self.a, self.b, self.cc = f(self.a), f(self.b), f(self.cc)
        self.v0, self.v_rst, self.v_th, self.poisson_weight = f(self.v0), f(self.v_rst), f(self.v_th), f(self.poisson_weight)

        # weights: sign(pre) * count * w_syn, CSR by presynaptic neuron
        W = c.W
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int32)
        pre_of_nz = np.repeat(np.arange(self.n), np.diff(W.indptr))
        self.w = (W.data.astype(np.float64) * c.sign[pre_of_nz] * self.w_syn).astype(dtype)

        # state
        self.v = np.full(self.n, self.v0, dtype=dtype)
        self.g = np.zeros(self.n, dtype=dtype)
        self.i_ext = np.zeros(self.n, dtype=dtype)
        self.rfc_left = np.zeros(self.n, dtype=np.int32)
        self.frozen = np.zeros(self.n, dtype=bool)      # refractory during the current step
        self.rfc_steps = np.full(self.n, self.rfc_default, dtype=np.int32)
        self.silenced = np.zeros(self.n, dtype=bool)
        self._spike_buf = np.zeros(self.n, dtype=np.int64)
        self.step_idx = 0
        self.queue: list[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(self.delay_steps)]
        self.poisson: dict[int, _PoissonInput] = {}
        self._next_handle = 0
        self.bg_indices: np.ndarray | None = None
        self.bg_p = 0.0
        self.bg_weight = 0.0
        self.subscriptions: list[Subscription] = []

        # backend
        if backend == "auto":
            backend = "torch" if (HAVE_TORCH and _torch.cuda.is_available()) else ("numba" if HAVE_NUMBA else "numpy")
        if backend == "numba" and not HAVE_NUMBA:
            backend = "numpy"
        self.backend = backend
        if backend == "torch":
            self._init_torch(device)

        bg = sim_cfg.get("background") or {}
        if bg.get("enabled"):
            tgt = None
            if bg.get("super_classes"):
                tgt = {"super_class": list(bg["super_classes"])}
            self.set_background(float(bg["rate_hz"]), float(bg["weight_mV"]), tgt)

    # ----------------------------------------------------------------- torch backend
    def _init_torch(self, device):
        t = _torch
        self.device = t.device(device or ("cuda" if t.cuda.is_available() else "cpu"))
        td = t.float32 if self.dtype == np.float32 else t.float64
        self.t_v = t.tensor(self.v, device=self.device)
        self.t_g = t.tensor(self.g, device=self.device)
        self.t_iext = t.tensor(self.i_ext, device=self.device)
        self.t_rfc_left = t.tensor(self.rfc_left, device=self.device)
        self.t_rfc_steps = t.tensor(self.rfc_steps, device=self.device)
        self.t_indptr = t.tensor(self.indptr, device=self.device)
        self.t_indices = t.tensor(self.indices.astype(np.int64), device=self.device)
        self.t_w = t.tensor(self.w, device=self.device, dtype=td)

    def _sync_from_torch(self):
        self.v = self.t_v.cpu().numpy(); self.g = self.t_g.cpu().numpy()
        self.rfc_left = self.t_rfc_left.cpu().numpy()

    # ----------------------------------------------------------------- input API
    def resolve(self, target, side=None) -> np.ndarray:
        return resolve_target(self.c, target, side=side)

    def inject_current(self, target, amp_mV: float, side=None) -> np.ndarray:
        """Constant input current, in mV of steady-state depolarisation (v -> v_0 + amp).

        Persistent until changed; ``amp_mV=0`` removes it. Returns the target indices.
        """
        idx = self.resolve(target, side)
        self.i_ext[idx] = amp_mV
        if self.backend == "torch":
            self.t_iext[_torch.as_tensor(idx, device=self.device)] = float(amp_mV)
        return idx

    def inject_poisson(self, target, rate_hz: float, weight_mV: float | None = None,
                       to: str = "v", refractory_free: bool = True, side=None) -> int:
        """Poisson spike input to a neuron set; returns a handle for :meth:`remove_input`.

        Default weight (``poisson_weight_mV``, 68.75 mV) onto ``v`` forces one spike per
        event, i.e. optogenetic-style activation exactly as in Shiu et al.  ``to='g'``
        delivers events to the synaptic conductance instead (weight in mV, like a synapse).
        With ``refractory_free`` the targets lose their refractory period (Shiu et al.).
        """
        idx = self.resolve(target, side)
        w = self.poisson_weight if weight_mV is None else float(weight_mV)
        if refractory_free:
            self.rfc_steps[idx] = 0
            if self.backend == "torch":
                self.t_rfc_steps[_torch.as_tensor(idx, device=self.device)] = 0
        h = self._next_handle; self._next_handle += 1
        self.poisson[h] = _PoissonInput(idx, float(rate_hz) * self.dt * 1e-3, w, to == "v")
        return h

    def set_poisson_rate(self, handle: int, rate_hz: float):
        self.poisson[handle].p = float(rate_hz) * self.dt * 1e-3

    def remove_input(self, handle: int):
        inp = self.poisson.pop(handle)
        self.rfc_steps[inp.indices] = self.rfc_default
        if self.backend == "torch":
            self.t_rfc_steps[_torch.as_tensor(inp.indices, device=self.device)] = self.rfc_default

    def set_background(self, rate_hz: float, weight_mV: float, target=None):
        """Independent Poisson background drive (events add ``weight_mV`` to g) on all
        neurons or on ``target``. ``rate_hz=0`` disables it."""
        if rate_hz <= 0:
            self.bg_indices = None; self.bg_p = 0.0; self.bg_weight = 0.0
            return
        self.bg_indices = np.arange(self.n) if target is None else self.resolve(target)
        self.bg_p = float(rate_hz) * self.dt * 1e-3
        self.bg_weight = float(weight_mV)

    def silence(self, target, side=None) -> np.ndarray:
        """Zero all output synapses of the target neurons (Shiu et al. 'silencing')."""
        idx = self.resolve(target, side)
        for i in idx:
            self.w[self.indptr[i]:self.indptr[i + 1]] = 0
        self.silenced[idx] = True
        if self.backend == "torch":
            self.t_w = _torch.tensor(self.w, device=self.device)
        return idx

    # ----------------------------------------------------------------- spike API
    def subscribe(self, target, name: str | None = None, callback: Callable | None = None,
                  side=None) -> Subscription:
        idx = self.resolve(target, side)
        member = np.zeros(self.n, dtype=bool); member[idx] = True
        s = Subscription(name or str(target), idx, member, callback)
        self.subscriptions.append(s)
        return s

    def unsubscribe(self, s: Subscription):
        self.subscriptions.remove(s)

    # ----------------------------------------------------------------- stepping
    def _poisson_events(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw this step's Poisson/background events. Returns (v_targets, g_targets, g_weights)."""
        v_t, g_t, g_w = [], [], []
        for inp in self.poisson.values():
            m = len(inp.indices)
            if m == 0 or inp.p <= 0:
                continue
            if inp.p * m < 8:   # sparse: draw count then targets (exact for Binomial thinning)
                k = self.rng.binomial(m, inp.p)
                if k == 0:
                    continue
                hit = inp.indices[self.rng.choice(m, k, replace=False)]
            else:
                hit = inp.indices[self.rng.random(m) < inp.p]
            if inp.to_v:
                v_t.append(hit)
            else:
                g_t.append(hit); g_w.append(np.full(len(hit), inp.weight, dtype=self.dtype))
        if self.bg_indices is not None and self.bg_p > 0:
            m = len(self.bg_indices)
            k = self.rng.binomial(m, self.bg_p)
            if k:
                hit = self.bg_indices[self.rng.integers(0, m, k)]
                g_t.append(hit); g_w.append(np.full(k, self.bg_weight, dtype=self.dtype))
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)
        return cat(v_t, np.int64), cat(g_t, np.int64), cat(g_w, self.dtype)

    def step(self) -> np.ndarray:
        """Advance one timestep; returns the indices of neurons that spiked in this step."""
        if self.backend == "torch":
            spikes = self._step_torch()
        elif self.backend == "numba":
            spikes = self._step_numba()
        else:
            spikes = self._step_numpy()
        for s in self.subscriptions:
            s.deliver(self.step_idx, spikes)
        self.step_idx += 1
        return spikes

    def _step_numba(self) -> np.ndarray:
        k = _nb_integrate(self.v, self.g, self.i_ext, self.rfc_left, self.frozen, self.v0, self.a, self.b,
                          self.cc, self.v_th, self._spike_buf)
        spikes = self._spike_buf[:k].copy()
        # synapses slot: delayed deliveries + Poisson events (dropped on refractory targets)
        due = self.queue.pop(0)
        if len(due):
            _nb_deliver(self.indptr, self.indices, self.w, due, self.g, self.frozen)
        v_t, g_t, g_w = self._poisson_events()
        if len(v_t):
            _nb_add_at(self.v, v_t, self.poisson_weight, self.frozen)
        if len(g_t):
            _nb_add_at_w(self.g, g_t, g_w, self.frozen)
        # reset slot
        if k:
            _nb_reset(self.v, self.g, self.rfc_left, self.rfc_steps, spikes, self.v_rst)
        self.queue.append(spikes)
        return spikes

    def _step_numpy(self) -> np.ndarray:
        active = self.rfc_left <= 0
        self.frozen = ~active
        self.rfc_left[~active] -= 1
        ss = self.v0 + self.i_ext
        vn = ss + self.a * (self.v - ss) + self.b * self.g
        self.v = np.where(active, vn, self.v).astype(self.dtype)
        self.g = np.where(active, self.cc * self.g, self.g).astype(self.dtype)
        spikes = np.flatnonzero(active & (self.v > self.v_th))
        due = self.queue.pop(0)
        if len(due):
            starts, ends = self.indptr[due], self.indptr[due + 1]
            lens = ends - starts
            tot = int(lens.sum())
            if tot:
                flat = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(tot)
                tgt = self.indices[flat]; keep = active[tgt]
                self.g += np.bincount(tgt[keep], weights=self.w[flat][keep], minlength=self.n).astype(self.dtype)
        v_t, g_t, g_w = self._poisson_events()
        if len(v_t):
            v_t = v_t[active[v_t]]
            np.add.at(self.v, v_t, self.poisson_weight)
        if len(g_t):
            keep = active[g_t]
            np.add.at(self.g, g_t[keep], g_w[keep])
        if len(spikes):
            self.v[spikes] = self.v_rst; self.g[spikes] = 0; self.rfc_left[spikes] = self.rfc_steps[spikes]
        self.queue.append(spikes)
        return spikes

    def _step_torch(self) -> np.ndarray:
        t = _torch
        active = self.t_rfc_left <= 0
        self.t_rfc_left.sub_((~active).to(self.t_rfc_left.dtype))
        ss = self.v0 + self.t_iext
        vn = ss + self.a * (self.t_v - ss) + self.b * self.t_g
        self.t_v = t.where(active, vn, self.t_v)
        self.t_g = t.where(active, self.cc * self.t_g, self.t_g)
        spk_t = t.nonzero(active & (self.t_v > self.v_th)).flatten()
        due = self.queue.pop(0)
        if len(due):
            due_t = t.as_tensor(due, device=self.device)
            starts = self.t_indptr[due_t]; lens = self.t_indptr[due_t + 1] - starts
            tot = int(lens.sum())
            if tot:
                off = t.cumsum(lens, 0) - lens
                flat = t.repeat_interleave(starts - off, lens) + t.arange(tot, device=self.device)
                tgt = self.t_indices[flat]
                self.t_g.index_add_(0, tgt, self.t_w[flat] * active[tgt].to(self.t_w.dtype))
        v_t, g_t, g_w = self._poisson_events()
        if len(v_t):
            vt = t.as_tensor(v_t, device=self.device)
            self.t_v.index_add_(0, vt, float(self.poisson_weight) * active[vt].to(self.t_v.dtype))
        if len(g_t):
            gt = t.as_tensor(g_t, device=self.device)
            self.t_g.index_add_(0, gt, t.as_tensor(g_w, device=self.device, dtype=self.t_g.dtype) * active[gt].to(self.t_g.dtype))
        if len(spk_t):
            self.t_v[spk_t] = float(self.v_rst); self.t_g[spk_t] = 0.0
            self.t_rfc_left[spk_t] = self.t_rfc_steps[spk_t]
        spikes = spk_t.cpu().numpy().astype(np.int64)
        self.queue.append(spikes)
        return spikes

    def run(self, duration_s: float | None = None, n_steps: int | None = None,
            record: bool = True, callback: Callable | None = None) -> SpikeRecord | None:
        """Run for ``duration_s`` (or ``n_steps``). Optionally record all spikes and call
        ``callback(step, spikes)`` each step."""
        if n_steps is None:
            n_steps = int(round(duration_s * 1e3 / self.dt))
        st, nu = [], []
        for _ in range(n_steps):
            s = self.step()
            if callback is not None:
                callback(self.step_idx - 1, s)
            if record and len(s):
                st.append(np.full(len(s), self.step_idx - 1, dtype=np.int64)); nu.append(s)
        if not record:
            return None
        if st:
            return SpikeRecord(self.dt, np.concatenate(st), np.concatenate(nu), n_steps)
        return SpikeRecord(self.dt, np.zeros(0, np.int64), np.zeros(0, np.int64), n_steps)

    @property
    def time_s(self) -> float:
        return self.step_idx * self.dt * 1e-3

    def reset_state(self, seed: int | None = None):
        """Return every neuron to rest and clear the delay queue (inputs stay registered)."""
        self.v[:] = self.v0; self.g[:] = 0; self.rfc_left[:] = 0
        self.queue = [np.zeros(0, dtype=np.int64) for _ in range(self.delay_steps)]
        self.step_idx = 0
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        if self.backend == "torch":
            self.t_v.fill_(self.v0); self.t_g.zero_(); self.t_rfc_left.zero_()
