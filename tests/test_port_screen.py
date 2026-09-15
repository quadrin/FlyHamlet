"""Development/validation separation, guard semantics, and end-to-end toy execution."""
from argparse import Namespace
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flyhamlet.activity import ActivityAbort, ActivityLimits, ActivityMonitor, GuardedNetwork
from flyhamlet.port_screen import (ScreenConfig, ScreenRejected, graph_hash, load_source,
    port_metrics, run_screen, select_opponent_groups)
from flyhamlet.prosthesis_memory import SIM, make_toy, schedule
from flyhamlet.screened_memory import run_benchmark, validate_manifest
from flyhamlet.sim import HAVE_NUMBA, HAVE_TORCH, LIFNetwork

BACKENDS = ["numpy"] + (["numba"] if HAVE_NUMBA else []) + (["torch"] if HAVE_TORCH else [])


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True])
def test_invalid_activity_limit(bad):
    with pytest.raises(ValueError):
        ActivityLimits(max_mean_hz=bad)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf, .3])
def test_invalid_monitor_timestep(dt):
    with pytest.raises(ValueError):
        ActivityMonitor(2, dt, ActivityLimits())


@pytest.mark.parametrize("bad", [dict(seed=-1), dict(group_size=0), dict(group_size=1.5),
    dict(development=1), dict(validation=0), dict(min_success=1.1), dict(min_success=np.nan),
    dict(drives_mV=()), dict(drives_mV=(9, 3)), dict(drives_mV=(3, 3)), dict(drives_mV=(np.inf,))])
def test_invalid_screen_config(bad):
    with pytest.raises(ValueError):
        ScreenConfig(**bad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_monitor_is_read_only_and_resets(backend):
    c, ports = make_toy()
    records = []
    a = GuardedNetwork(c, SIM, backend=backend, device="cpu", activity_records=records)
    b = LIFNetwork(c, SIM, backend=backend, device="cpu")
    for net in (a, b):
        net.inject_poisson(ports["cue"][0], 100)
    for seed in (11, 12):
        a.reset_state(seed); b.reset_state(seed)
        ra, rb = a.run(duration_s=.1), b.run(duration_s=.1)
        np.testing.assert_array_equal(ra.steps, rb.steps)
        np.testing.assert_array_equal(ra.neurons, rb.neurons)
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
        if backend == "torch":
            a._sync_from_torch(); b._sync_from_torch()
        for name in ("v", "g", "rfc_left", "w", "i_ext"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    a.finish_trial()
    assert len(records) == 2 and all(len(r["windows"]) == 2 for r in records)
    with pytest.raises(RuntimeError, match="reset_state"):
        a.step()


@pytest.mark.parametrize("kind", ["mean", "cell", "group", "nonfinite"])
def test_guard_reasons_and_latch(kind):
    net = SimpleNamespace(backend="numpy", v=np.zeros(2), g=np.zeros(2), seed=123)
    limits = ActivityLimits(window_ms=10, max_mean_hz=10000, max_cell_hz=10000, max_group_hz=10000)
    key = {"mean": "max_mean_hz", "cell": "max_cell_hz", "group": "max_group_hz"}.get(kind)
    if key:
        limits = replace(limits, **{key: 1})
    else:
        net.v[0] = np.nan
    mon = ActivityMonitor(2, 1.0, limits, {"small": [0]})
    for _ in range(9):
        mon.observe(net, np.array([0]))
    with pytest.raises(ActivityAbort):
        mon.observe(net, np.array([0]))
    assert mon.aborted and mon.aborted["noise_seed"] == 123
    with pytest.raises(ActivityAbort):
        mon.observe(net, np.array([], dtype=int))


def test_partial_window_is_checked():
    net = SimpleNamespace(backend="numpy", v=np.zeros(1), g=np.zeros(1), seed=1)
    mon = ActivityMonitor(1, 1.0, ActivityLimits(max_cell_hz=2000, max_mean_hz=2000))
    mon.observe(net, np.array([0]))
    mon.flush(net)
    assert mon.windows[0]["duration_ms"] == 1
    assert mon.windows[0]["mean_hz"] == 1000


def test_reset_recovers_aborted_network_explicitly():
    c, _ = make_toy()
    net = GuardedNetwork(c, SIM, backend="numpy", limits=ActivityLimits(max_mean_hz=.001))
    net.inject_current({"index": [0]}, 20)
    with pytest.raises(ActivityAbort):
        net.run(duration_s=.1)
    at = net.step_idx
    with pytest.raises(ActivityAbort):
        net.step()
    assert net.step_idx == at
    net.inject_current({"index": [0]}, 0)
    net.reset_state()
    net.run(duration_s=.1)
    assert net.activity.aborted is None


def test_opponent_selection_is_disjoint_and_stable():
    groups = select_opponent_groups(np.array([3, 3, -4, -4, 0]), np.arange(5), [0], 1, 1)
    assert [g.tolist() for g in groups] == [[1], [2]]
    with pytest.raises(ScreenRejected):
        select_opponent_groups(np.zeros(5), np.arange(5), [], 2, 1)


@pytest.fixture(scope="module")
def screened(tmp_path_factory):
    out = tmp_path_factory.mktemp("screen")
    c, sim, specs = load_source(True, Path("unused"))
    cfg = ScreenConfig(development=2, validation=2)
    report = run_screen(c, sim, specs, cfg, ActivityLimits(), "numpy", out, toy=True)
    return c, sim, specs, report, out


def test_screen_passes_fresh_seeds_and_exports_root_ids(screened):
    c, sim, specs, report, out = screened
    assert report["status"] == "validated" and report["drive_mV"] == 9
    assert not set(report["development_noise_seeds"]) & set(report["validation_noise_seeds"])
    ports = json.loads((out / "ports.json").read_text())
    indices = validate_manifest(report, ports, c, sim, toy=True, schedules=[(8, 101, 30)])
    flat = np.concatenate([g for groups in indices.values() for g in groups])
    assert len(flat) == len(np.unique(flat))
    assert all(min(g["ids"]) >= 2**40 for groups in ports.values() for g in groups)
    assert all(v["success"] == 1 for k, v in report["validation"].items() if k != "passed")


def test_empty_existing_output_refused(screened):
    c, sim, specs, _, out = screened
    with pytest.raises(ValueError, match="output must be empty"):
        run_screen(c, sim, specs, ScreenConfig(development=2, validation=2), ActivityLimits(), "numpy", out)


def test_original_weak_feedback_toy_is_rejected(tmp_path):
    c, _ = make_toy()
    _, sim, specs = load_source(True, Path("unused"))
    with pytest.raises(ScreenRejected):
        run_screen(c, sim, specs, ScreenConfig(development=2, validation=2), ActivityLimits(), "numpy", tmp_path)
    assert not (tmp_path / "ports.json").exists()
    assert json.loads((tmp_path / "screen.json").read_text())["status"] == "rejected"


def test_guard_abort_exports_failure_not_ports(tmp_path):
    c, sim, specs = load_source(True, Path("unused"))
    with pytest.raises(ActivityAbort):
        run_screen(c, sim, specs, ScreenConfig(development=2, validation=2),
                   ActivityLimits(max_mean_hz=.001), "numpy", tmp_path)
    result = json.loads((tmp_path / "screen.json").read_text())
    assert result["status"] == "aborted" and result["failed_trial_windows"]
    assert not (tmp_path / "ports.json").exists()


def test_validation_failure_does_not_try_second_candidate(tmp_path, monkeypatch):
    import flyhamlet.port_screen as module
    c, sim, specs = load_source(True, Path("unused"))
    specs["cue_pairs"] *= 2
    calls = []
    original = module.port_metrics
    def reject_validation(*a, **kw):
        result = original(*a, **kw)
        calls.append(result)
        if len(calls) > 1:
            result["passed"] = False
        return result
    monkeypatch.setattr(module, "port_metrics", reject_validation)
    with pytest.raises(ScreenRejected, match="fresh-seed"):
        run_screen(c, sim, specs, ScreenConfig(development=2, validation=2), ActivityLimits(), "numpy", tmp_path)
    report = json.loads((tmp_path / "screen.json").read_text())
    assert len(calls) == 2 and len(report["attempts"]) == 1
    assert not (tmp_path / "ports.json").exists()


@pytest.mark.parametrize("field", ["graph_sha256", "sim_sha256", "ports_sha256", "source_sha256", "status", "toy"])
def test_stale_or_mislabeled_manifest_rejected(screened, field):
    c, sim, _, report, out = screened
    manifest = copy.deepcopy(report)
    manifest[field] = "incorrect"
    with pytest.raises(ValueError):
        validate_manifest(manifest, json.loads((out / "ports.json").read_text()), c, sim, toy=True, schedules=[])


def test_noise_reuse_and_graph_mutation_rejected(screened):
    c, sim, _, report, out = screened
    ports = json.loads((out / "ports.json").read_text())
    reused = copy.deepcopy(report)
    reused["development_noise_seeds"].append(int(schedule(2, 101, 10)[1][0]))
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(reused, ports, c, sim, toy=True, schedules=[(2, 101, 10)])
    changed = copy.deepcopy(c)
    changed.W.data[0] += 1
    assert graph_hash(changed) != graph_hash(c)
    with pytest.raises(ValueError, match="graph_sha256"):
        validate_manifest(report, ports, changed, sim, toy=True, schedules=[])


def benchmark_args(screened, out):
    folder = screened[-1]
    return Namespace(toy=True, config=Path("unused"), ports=folder / "ports.json",
          manifest=folder / "screen.json", out=out, backend="numpy", seeds=[101], units=8,
          delay_ms=500.0, calibration=2, train=2, evaluate=4,
          conditions=["native", "read_only", "memoryless", "leaky", "closed_loop", "disconnected"])


def test_guarded_benchmark_end_to_end(screened, tmp_path):
    report = run_benchmark(benchmark_args(screened, tmp_path))
    assert report["status"] == "completed" and len(report["results"]) == 6
    assert len(report["paired_seed_differences"]) == 5
    for result in report["results"]:
        records = json.loads((tmp_path / "seed_101" / (result["condition"] + "_activity.json")).read_text())
        expected = 6 if result["condition"] == "native" else 8
        assert len(records) == expected
        assert all(r["n_steps"] == 7000 for r in records)
        assert all(len(r["windows"]) == 14 for r in records)
    assert (tmp_path / "summary.json").exists()


def test_benchmark_abort_has_no_success_summary(screened, tmp_path):
    args = benchmark_args(screened, tmp_path / "run")
    manifest = copy.deepcopy(screened[3])
    manifest["limits"]["max_mean_hz"] = .001
    args.manifest = tmp_path / "strict.json"
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ActivityAbort):
        run_benchmark(args)
    assert not (args.out / "summary.json").exists()
    assert json.loads((args.out / "status.json").read_text())["status"] == "aborted"
