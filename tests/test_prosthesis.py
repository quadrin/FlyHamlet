"""No FlyWire download: integration tests use the repository's actual LIF engine."""
from dataclasses import replace
import copy
import numpy as np
import pytest

from flyhamlet.sim import LIFNetwork, HAVE_NUMBA, HAVE_TORCH
from flyhamlet.prosthesis import ProsthesisConfig, RecurrentProsthesis, fit_ridge
from flyhamlet.prosthesis_memory import make_toy, SIM, schedule, validate_ports, wilson

BACKENDS = ["numpy"] + (["numba"] if HAVE_NUMBA else []) + (["torch"] if HAVE_TORCH else [])


def setup(backend="numpy", **kwargs):
    c, ports = make_toy()
    net = LIFNetwork(c, SIM, seed=44, backend=backend, device="cpu")
    config = ProsthesisConfig(n_units=16, **kwargs)
    lobe = RecurrentProsthesis(c.n, net.dt, ports["read"], ports["write"], config)
    return net, lobe, ports


def state(net):
    if net.backend == "torch":
        net._sync_from_torch()
    return net.v.copy(), net.g.copy(), net.rfc_left.copy()


@pytest.mark.parametrize("kwargs", [dict(n_units=0), dict(n_units=1.5), dict(n_units=True),
    dict(update_ms=0), dict(tau_ms=-1), dict(rate_scale_hz=np.nan), dict(seed=-1),
    dict(recurrent_gain=1), dict(recurrent_gain=-.1), dict(max_drive_mV=np.inf), dict(mode="invalid")])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ProsthesisConfig(**kwargs)


@pytest.mark.parametrize("groups", [[], [[]], [[-1]], [[64]], [[1.5]], [[1,1]], [[1], [1]], [[True]]])
def test_bad_ports(groups):
    with pytest.raises(ValueError):
        RecurrentProsthesis(64, .1, groups, [[3]], ProsthesisConfig(n_units=4))


def test_fractional_sampling_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        RecurrentProsthesis(4, .3, [[0]], [[2]])


def test_pool_rates_and_timing():
    p = RecurrentProsthesis(4, 1, [[0, 1]], [[2]], ProsthesisConfig(n_units=8))
    empty = np.array([], dtype=np.int64)
    assert not p.observe(np.array([0]))
    for _ in range(8):
        assert not p.observe(empty)
    assert p.n_updates == 0 and not p.state.any()
    assert p.observe(empty)
    assert p.n_updates == 1
    assert p.last_rates_hz.tolist() == [50.0]
    assert p.state.any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_feedback_preserves_spikes_and_rng(backend):
    a, lobe, ports = setup(backend)
    b = LIFNetwork(a.c, SIM, seed=44, backend=backend, device="cpu")
    before_rng = copy.deepcopy(a.rng.bit_generator.state)
    before_w = a.w.copy()
    a.attach_prosthesis(lobe)
    assert before_rng == a.rng.bit_generator.state
    for net in (a, b):
        net.inject_poisson(ports["cue"][0], 300)
    ra, rb = a.run(duration_s=.05), b.run(duration_s=.05)
    np.testing.assert_array_equal(ra.steps, rb.steps)
    np.testing.assert_array_equal(ra.neurons, rb.neurons)
    for x,y in zip(state(a), state(b)):
        np.testing.assert_array_equal(x,y)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    np.testing.assert_array_equal(before_w, a.w)
    assert lobe.state.any() and not lobe.drive_mV.any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_feedback_is_causal_and_clipped(backend):
    net, lobe, ports = setup(backend)
    lobe.bias[:] = 1000  # even manually altered outputs remain bounded
    net.attach_prosthesis(lobe)
    net.run(n_steps=lobe.bin_steps, record=False)
    assert np.all(state(net)[1] == 0)  # first complete bin did not affect its own spikes
    assert np.all(lobe.drive_mV == lobe.config.max_drive_mV)
    net.step()
    expected = (1.0 - float(net.cc)) * lobe.config.max_drive_mV
    np.testing.assert_allclose(state(net)[1][lobe.write_indices], expected, atol=1e-7)
    rec = net.run(duration_s=.15)
    assert not len(rec.neurons)  # 3 mV steady drive stays below the resting threshold gap
    np.testing.assert_allclose(state(net)[1][lobe.write_indices], 3, atol=1e-4)
    assert lobe.saturated_outputs > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_feedback_discarded_during_refractory_and_spike_reset(backend):
    net, lobe, ports = setup(backend)
    net.attach_prosthesis(lobe)
    lobe.drive_mV[:] = 3
    target = lobe.write_indices[0]
    if backend == "torch":
        net.t_rfc_left[target] = 3
    else:
        net.rfc_left[target] = 3
    net.step()
    assert state(net)[1][target] == 0
    if backend == "torch":
        net.t_rfc_left[target] = 0
        net.t_v[target] = 0
    else:
        net.rfc_left[target] = 0
        net.v[target] = 0
    assert target in net.step()
    assert state(net)[1][target] == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_clears_history_and_reproduces_trials(backend):
    net, lobe, ports = setup(backend, max_drive_mV=12)
    net.attach_prosthesis(lobe)
    lobe.w_out[:] = 1
    net.inject_poisson(ports["cue"][0], 150)
    ra = net.run(duration_s=.04)
    digest = lobe.weights_digest()
    ha = lobe.state.copy()
    assert lobe.state.any()
    net.run(n_steps=3, record=False)
    assert lobe._ticks == 3
    net.reset_state(seed=44)
    assert not lobe.state.any() and not lobe._counts.any()
    assert not lobe.drive_mV.any() and not lobe.last_rates_hz.any()
    assert lobe._ticks == 0 and lobe.n_updates == 0
    assert digest == lobe.weights_digest()
    rb = net.run(duration_s=.04)
    np.testing.assert_array_equal(ra.steps,rb.steps)
    np.testing.assert_array_equal(ra.neurons,rb.neurons)
    np.testing.assert_array_equal(ha,lobe.state)


def test_feedback_does_not_overwrite_currents_or_synapses():
    net, lobe, ports = setup()
    net.inject_current(ports["write"][0], 1.2)
    current, weights = net.i_ext.copy(), net.w.copy()
    net.attach_prosthesis(lobe)
    lobe.bias[:] = 2
    net.run(duration_s=.03)
    np.testing.assert_array_equal(current, net.i_ext)
    np.testing.assert_array_equal(weights,net.w)
    old_g = net.g.copy()
    assert net.detach_prosthesis() is lobe
    assert lobe._owner is None
    assert np.array_equal(old_g,net.g)  # no retroactive removal of delivered g
    net.step()
    assert np.all(net.g <= old_g)
    np.testing.assert_array_equal(current,net.i_ext)


def test_exclusive_attachment_and_mismatch():
    net, lobe, _ = setup()
    net2, _, _ = setup()
    net.attach_prosthesis(lobe)
    with pytest.raises(RuntimeError):
        net.attach_prosthesis(lobe)
    with pytest.raises(RuntimeError):
        net2.attach_prosthesis(lobe)
    net.detach_prosthesis()
    net2.attach_prosthesis(lobe)
    bad = RecurrentProsthesis(4, .1, [[0]], [[1]])
    with pytest.raises(ValueError):
        net.attach_prosthesis(bad)
    with pytest.raises(TypeError):
        net.attach_prosthesis(object())


def test_checkpoint_and_training(tmp_path):
    _, lobe, _ = setup()
    rng = np.random.default_rng(19)
    states = rng.normal(size=(100,16)) * .1
    targets = np.column_stack([states[:,0], states[:,1]])
    w_in, w_rec = lobe.w_in.copy(), lobe.w_rec.copy()
    assert lobe.fit_feedback(states, targets, ridge=1e-6) < 1e-6
    np.testing.assert_array_equal(lobe.w_in,w_in)
    np.testing.assert_array_equal(lobe.w_rec,w_rec)
    digest = lobe.weights_digest()
    path = tmp_path / 'weights.npz'
    lobe.save_weights(path)
    lobe.w_out[:] = 0
    lobe.state[:] = 1
    lobe.load_weights(path)
    assert lobe.weights_digest() == digest and not lobe.state.any()
    _, other, _ = setup(seed=7)
    with pytest.raises(ValueError,match="configuration"):
        other.load_weights(path)
    with pytest.raises(ValueError):
        lobe.fit_feedback(states, np.full((100,2),4.0))


def test_bad_checkpoint_is_atomic(tmp_path):
    _,lobe,_=setup()
    path=tmp_path/'bad.npz'
    lobe.save_weights(path)
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    d['w_rec'][:]=np.nan
    np.savez(path,**d)
    before=lobe.weights_digest()
    with pytest.raises(ValueError):
        lobe.load_weights(path)
    assert before==lobe.weights_digest()


@pytest.mark.parametrize("n,p", [(8,16),(30,4)])
def test_ridge_primal_and_dual(n,p):
    rng=np.random.default_rng(2)
    x=rng.normal(size=(n,p));y=rng.normal(size=(n,2))
    w,b=fit_ridge(x,y,.2)
    xc=x-x.mean(0);yc=y-y.mean(0)
    expected=np.linalg.solve(xc.T@xc+.2*np.eye(p),xc.T@yc).T
    np.testing.assert_allclose(w,expected,atol=1e-12)
    np.testing.assert_allclose(b,y.mean(0)-w@x.mean(0),atol=1e-12)


def test_ablations_remove_exact_memory_sources():
    common=ProsthesisConfig(n_units=8,update_ms=1,tau_ms=100)
    modules=[RecurrentProsthesis(4,1,[[0]],[[2]],replace(common,mode=mode))
             for mode in ('recurrent','leaky','memoryless')]
    for lobe in modules:
        lobe.observe(np.array([0],dtype=np.int64))
        for _ in range(20):
            lobe.observe(np.array([],dtype=np.int64))
        assert np.max(np.abs(lobe.state)) <= 1
    assert modules[0].state.any() and modules[1].state.any()
    assert not modules[2].state.any()
    assert not np.array_equal(modules[0].state,modules[1].state)
    assert not modules[1].w_rec.any() and not modules[2].w_rec.any()
    np.testing.assert_array_equal(modules[0].w_in,modules[2].w_in)


def test_schedule_disjoint_phases_and_balanced():
    a,seeds_a=schedule(20,6,10)
    b,seeds_b=schedule(20,6,30)
    assert a.sum()==b.sum()==10
    assert not set(seeds_a).intersection(seeds_b)
    np.testing.assert_array_equal(a,schedule(20,6,10)[0])
    with pytest.raises(ValueError):
        schedule(3,6,0)


def test_protocol_prevents_sensory_and_stimulation_shortcuts():
    c,ports=make_toy()
    assert validate_ports(c,ports)
    ports['report']=ports['write']
    with pytest.raises(ValueError,match='disjoint'):
        validate_ports(c,ports)


def test_nonfinite_feedback_fails():
    net,lobe,_=setup()
    net.attach_prosthesis(lobe)
    lobe.drive_mV[0]=np.nan
    with pytest.raises(FloatingPointError):
        net.step()


def test_closed_loop_backend_parity():
    records={}
    for backend in BACKENDS:
        net,lobe,ports=setup(backend,max_drive_mV=12)
        net.attach_prosthesis(lobe)
        lobe.w_out[0,:]=30
        lobe.w_out[1,:]=-30
        net.inject_poisson(ports['cue'][0],300)
        rec=net.run(duration_s=.06)
        records[backend]=(rec.steps,rec.neurons)
    for backend in BACKENDS:
        for x,y in zip(records['numpy'],records[backend]):
            np.testing.assert_array_equal(x,y)


def test_bin_reset_erases_cue_and_observation_cut():
    _,lobe,ports=setup()
    for _ in range(3):
        lobe.observe(ports['read'][0])
    assert lobe._counts.any()
    lobe.reset_state()
    lobe.observation_enabled=False
    for _ in range(lobe.bin_steps):
        lobe.observe(ports['read'][0])
    assert not lobe.state.any() and not lobe.last_rates_hz.any()


def test_wilson_intervals():
    assert wilson(0,20)[0]==0
    lo,hi=wilson(20,20)
    assert .83<lo<.85 and hi==1
