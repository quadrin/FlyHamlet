"""Simulator unit tests on a tiny synthetic connectome (no data download needed)."""
import math
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyhamlet.data import Connectome
from flyhamlet.sim import LIFNetwork, resolve_target

SIM = dict(dt_ms=0.1, v_rest_mV=-52.0, v_reset_mV=-52.0, v_thresh_mV=-45.0, tau_mem_ms=20.0, tau_syn_ms=5.0,
           t_refractory_ms=2.2, delay_ms=1.8, w_syn_mV=0.275, poisson_weight_mV=68.75, seed=0, dtype="float32",
           background={"enabled": False})


def toy(n=6, edges=((0, 1, 10), (1, 2, 40), (3, 2, 40), (2, 4, 1)), signs=(1, 1, 1, -1, 1, 1)):
    pre, post, cnt = zip(*edges)
    W = sp.csr_matrix((np.array(cnt, np.int32), (pre, post)), shape=(n, n))
    ids = np.arange(n) + (1 << 50)
    ct = (["A", "B", "C", "A", "MN", None] * n)[:n]
    neurons = pd.DataFrame({"root_id": ids, "cell_type": ct, "hemibrain_type": [None] * n,
                            "consolidated_type": [x or "X" for x in ct], "super_class": ["central"] * n,
                            "side": (["left", "right"] * n)[:n], "class": [None] * n,
                            "sub_class": [None] * n, "nerve": [None] * n, "nt": ["ACH"] * n, "sign": list(signs)})
    return Connectome(root_ids=ids, W=W, nt=np.array(["ACH"] * n), sign=np.array(signs, np.int8), neurons=neurons,
                      labels=pd.DataFrame({"root_id": [ids[min(4, n - 1)]], "label": ["motor neuron 9; MN9"], "user_name": ["t"]}))


@pytest.mark.parametrize("backend", ["numpy", "numba", "torch"])
def test_no_input_is_silent_and_deterministic(backend):
    c = toy()
    net = LIFNetwork(c, SIM, backend=backend)
    rec = net.run(duration_s=0.05)
    assert len(rec.neurons) == 0
    assert np.allclose(net.v, SIM["v_rest_mV"]) and np.allclose(net.g, 0)


@pytest.mark.parametrize("backend", ["numpy", "numba", "torch"])
def test_single_synapse_psp_matches_analytic(backend):
    """One forced spike in neuron 0 -> neuron 1 receives g = 10 * w_syn after the delay and
    v follows the exact alpha-like solution of the two linear ODEs."""
    c = toy()
    net = LIFNetwork(c, SIM, backend=backend)
    h = net.inject_poisson({"index": [0]}, rate_hz=0.0)          # registers, no events
    net.poisson[h].p = 0.0
    # force one event by hand at step 0 (weight forces spike at step 1)
    if backend == "torch":
        net.t_v[0] = SIM["v_rest_mV"] + 68.75
    else:
        net.v[0] = SIM["v_rest_mV"] + 68.75
    s1 = net.step(); assert list(s1) == [0]
    D = net.delay_steps
    for _ in range(D):
        net.step()
    if backend == "torch":
        net._sync_from_torch()
    g_expected = 10 * SIM["w_syn_mV"]
    assert math.isclose(float(net.g[1]), g_expected, rel_tol=1e-5)
    # integrate 20 more steps and compare with closed form u(t)=A(e^{-t/tm}-e^{-t/ts}), A=g0*ts/(ts-tm)... sign handled
    tm, ts, dt = 20.0, 5.0, 0.1
    g0 = g_expected
    v_before = float(net.v[1])
    for _ in range(20):
        net.step()
    if backend == "torch":
        net._sync_from_torch()
    t = 20 * dt
    A = g0 * ts / (ts - tm)
    u = (v_before - SIM["v_rest_mV"] - A) * math.exp(-t / tm) + A * math.exp(-t / ts)
    assert math.isclose(float(net.v[1]) - SIM["v_rest_mV"], u, rel_tol=1e-4, abs_tol=1e-4)
    assert len(net.run(duration_s=0.02).neurons) == 0     # 2.75 mV PSP is below the 7 mV threshold


def test_strong_synapse_fires_and_refractory_holds():
    c = toy()
    net = LIFNetwork(c, SIM, backend="numpy")
    net.inject_poisson({"index": [1]}, rate_hz=5000.0)       # neuron 1 spikes almost every step (rfc = 0)
    rec = net.run(duration_s=0.1)
    r = rec.rates_hz(c.n)
    assert r[1] > 2000 and r[2] > 0 and r[4] == 0            # 40 syn * 0.275 = 11 mV > threshold; C->MN too weak
    # refractory (Brian2 semantics): 21 frozen steps, so ISI >= 22 steps
    st = rec.steps[rec.neurons == 2]
    assert np.all(np.diff(st) >= 22)


def test_inhibition_sign():
    c = toy()
    net = LIFNetwork(c, SIM, backend="numpy")
    net.inject_poisson({"index": [3]}, rate_hz=5000.0)       # neuron 3 is inhibitory (sign -1) onto 2
    net.run(duration_s=0.05)
    assert net.g[2] < 0 and net.v[2] < SIM["v_rest_mV"]


def test_seed_reproducibility_and_backend_agreement():
    c = toy()
    outs = {}
    for backend in ["numpy", "numba", "torch"]:
        net = LIFNetwork(c, SIM, seed=123, backend=backend)
        net.inject_poisson({"type": "A"}, rate_hz=300.0)
        net.set_background(200.0, 0.5)
        rec = net.run(duration_s=0.2)
        outs[backend] = (rec.steps, rec.neurons)
    net = LIFNetwork(c, SIM, seed=123, backend="numpy")
    net.inject_poisson({"type": "A"}, rate_hz=300.0); net.set_background(200.0, 0.5)
    rec2 = net.run(duration_s=0.2)
    assert np.array_equal(rec2.steps, outs["numpy"][0]) and np.array_equal(rec2.neurons, outs["numpy"][1])
    assert len(rec2.neurons) > 10
    for b in ["numba", "torch"]:
        assert np.array_equal(outs[b][0], outs["numpy"][0]) and np.array_equal(outs[b][1], outs["numpy"][1]), b
    net3 = LIFNetwork(c, SIM, seed=124, backend="numpy")
    net3.inject_poisson({"type": "A"}, rate_hz=300.0); net3.set_background(200.0, 0.5)
    rec3 = net3.run(duration_s=0.2)
    assert not (np.array_equal(rec3.steps, rec2.steps) and np.array_equal(rec3.neurons, rec2.neurons))


def test_subscription_and_targets():
    c = toy()
    net = LIFNetwork(c, SIM, backend="numpy")
    assert list(resolve_target(c, "A")) == [0, 3]
    assert list(resolve_target(c, {"type": "A", "side": "right"})) == [3]
    assert list(resolve_target(c, {"label": "MN9"})) == [4]
    assert list(resolve_target(c, [1 << 50, (1 << 50) + 2])) == [0, 2]
    with pytest.raises(KeyError):
        resolve_target(c, "NoSuchType")
    seen = []
    sub = net.subscribe("B", callback=lambda step, s: seen.append((step, list(s))))
    net.inject_poisson({"index": [1]}, rate_hz=5000.0)
    net.run(duration_s=0.01)
    assert sub.total > 0 and all(s == [1] for _, s in seen if s)


def test_current_injection():
    c = toy()
    net = LIFNetwork(c, SIM, backend="numpy")
    net.inject_current({"index": [5]}, amp_mV=6.0)           # steady state -46 < threshold: no spike
    rec = net.run(duration_s=0.2)
    assert len(rec.neurons) == 0 and abs(net.v[5] - (-46.0)) < 0.05
    net.inject_current({"index": [5]}, amp_mV=8.0)           # steady state -44 > threshold: tonic firing
    rec = net.run(duration_s=0.2)
    assert (rec.neurons == 5).sum() >= 3


# ----------------------------------------------------------------------------- Brian2 reference
# Spike steps recorded from Brian2 2.9.0 running the Shiu et al. equations (method='linear',
# dt = 0.1 ms, delay 1.8 ms, refractory 2.2 ms) on two small deterministic chains; see
# tests/data/brian2_chain_*.json. Our backends must reproduce them spike for spike.
import json
from pathlib import Path

CHAINS = {
    "weak": dict(edges=((0, 1, 40), (1, 2, 25), (1, 3, 8), (2, 3, 12)), signs=(1, 1, -1, 1)),
    "strong": dict(edges=((0, 1, 200), (0, 2, 30), (1, 2, 60), (1, 3, 300), (2, 3, 50)), signs=(1, 1, -1, 1)),
}


@pytest.mark.parametrize("backend", ["numpy", "numba", "torch"])
@pytest.mark.parametrize("chain", ["weak", "strong"])
def test_matches_brian2_reference_chain(backend, chain):
    ref = json.load(open(Path(__file__).parent / "data" / f"brian2_chain_{chain}.json"))
    c = toy(n=4, **CHAINS[chain])
    sim = dict(SIM, dtype="float64")
    net = LIFNetwork(c, sim, backend=backend)
    net.inject_current({"index": [0]}, 100.0)
    rec = net.run(duration_s=0.3)
    for k in range(4):
        assert rec.steps[rec.neurons == k].tolist() == ref.get(str(k), []), f"neuron {k}"
