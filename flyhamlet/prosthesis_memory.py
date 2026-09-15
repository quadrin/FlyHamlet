"""Experiment 6 prototype: delayed binary cue, disjoint ports, explicit ablations.

python -m flyhamlet.prosthesis_memory --toy --out results/prosthesis

The toy graph tests the software; it supplies NO FlyWire/animal evidence.
For the actual connectome supply --config config.yaml --ports ports.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import Connectome
from .sim import LIFNetwork, resolve_target, HAVE_NUMBA, HAVE_TORCH
from .prosthesis import ProsthesisConfig, RecurrentProsthesis, fit_ridge, _validate_groups

SIM = dict(dt_ms=0.1, v_rest_mV=-52.0, v_reset_mV=-52.0, v_thresh_mV=-45.0,
           tau_mem_ms=20.0, tau_syn_ms=5.0, t_refractory_ms=2.2, delay_ms=1.8,
           w_syn_mV=0.275, poisson_weight_mV=68.75, seed=0, dtype="float32",
           background={"enabled": False})
CONDITIONS = ("native", "read_only", "memoryless", "leaky", "closed_loop", "disconnected")


def make_toy() -> tuple[Connectome, dict]:
    """Engineered 64-cell routing graph with no autonomous recurrent loops.

    Two cue -> relay pathways; two write -> reporter pathways. Weak write ->
    relay edges close the interface loop without intentionally building a latch.
    Every port uses different cells. All cells are artificial and excitatory.
    """
    size, n = 8, 64
    groups = [np.arange(i * size, (i + 1) * size, dtype=np.int64) for i in range(8)]
    ports = dict(cue=groups[:2], read=groups[2:4], write=groups[4:6], report=groups[6:8])
    edges = []
    for side in range(2):
        for source, target, count in (("cue", "read", 40), ("read", "report", 35),
                                      ("write", "read", 1), ("write", "report", 100)):
            edges.extend((i, j, count) for i in ports[source][side] for j in ports[target][side])
    pre, post, weights = np.array(edges).T
    w = sp.csr_matrix((weights.astype(np.int32), (pre, post)), shape=(n, n))
    ids = np.arange(n, dtype=np.int64) + (1 << 50)
    names = [f"toy_{key}_{side}" for key in ("cue", "read", "write", "report")
             for side in ("left", "right") for _ in range(size)]
    neurons = pd.DataFrame(dict(root_id=ids, cell_type=names, hemibrain_type=names,
                                 consolidated_type=names, side=[s for _ in range(4)
                                 for s in (["left"] * size + ["right"] * size)],
                                 super_class=["central"] * n))
    c = Connectome(ids, w, np.array(["ACH"] * n), np.ones(n, dtype=np.int8), neurons)
    return c, ports


def validate_ports(c: Connectome, ports: dict) -> dict:
    required = {"cue", "read", "write", "report"}
    if set(ports) != required:
        raise ValueError("ports must contain exactly cue, read, write, report")
    resolved = {name: _validate_groups([resolve_target(c, spec) for spec in specs], c.n, name)
                for name, specs in ports.items()}
    if len(resolved["cue"]) != 2 or len(resolved["write"]) != 2:
        raise ValueError("binary benchmark requires two cue groups and two write groups")
    # Prevent direct reading of the cue or direct stimulation of scored neurons.
    all_idx = np.concatenate([g for groups in resolved.values() for g in groups])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("benchmark cue/read/write/report ports must all be disjoint")
    return resolved


def schedule(n: int, seed: int, phase: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or n % 2:
        raise ValueError("trial counts must be even and >= 2")
    # Label order and sensory randomness never share an RNG stream.
    label_rng = np.random.default_rng(np.random.SeedSequence([seed, phase, 0]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, phase, 1]))
    labels = label_rng.permutation(np.tile([0, 1], n // 2))
    noise_seeds = noise_rng.integers(0, 2**32, size=n, dtype=np.uint64)
    return labels, noise_seeds


def steps_for(ms: float, dt: float) -> int:
    ratio = ms / dt
    if not np.isfinite(ms) or ms <= 0 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
        raise ValueError("durations must be positive integer multiples of the timestep")
    return int(round(ratio))


def trial(net, lobe, ports, handles, label, noise_seed, cue_ms, delay_ms, probe_ms, rate_hz):
    """Environment supplies a cue; the lobe itself sees only propagated spikes."""
    net.reset_state(seed=int(noise_seed))
    cue_steps = steps_for(cue_ms, net.dt)
    delay_steps = steps_for(delay_ms, net.dt)
    probe_steps = steps_for(probe_ms, net.dt)
    for i, handle in enumerate(handles):
        net.set_poisson_rate(handle, rate_hz if i == int(label) else 0.0)
    net.run(n_steps=cue_steps, record=False)
    for handle in handles:
        net.set_poisson_rate(handle, 0.0)
    net.run(n_steps=delay_steps, record=False)
    channel = np.full(net.n, -1, dtype=np.int32)
    for i, group in enumerate(ports["report"]):
        channel[group] = i
    counts = np.zeros(len(ports["report"]), dtype=np.int64)
    probe_spikes = 0
    for _ in range(probe_steps):
        spikes = net.step()
        probe_spikes += len(spikes)
        selected = channel[spikes]
        selected = selected[selected >= 0]
        if selected.size:
            counts += np.bincount(selected, minlength=len(counts))
    rates = counts / (np.array([len(g) for g in ports["report"]]) * probe_ms * 1e-3)
    features = np.clip(rates / 100.0, 0, 1)
    state = None if lobe is None else lobe.state.copy()
    stats = dict(report_rates_hz=rates.tolist(), probe_total_spikes=int(probe_spikes),
                 peak_prosthetic_drive_mV=0.0 if lobe is None else lobe.peak_abs_drive_mV,
                 saturated_outputs=0 if lobe is None else lobe.saturated_outputs)
    return features, state, stats


def wilson(correct: int, total: int) -> list[float]:
    p, z = correct / total, 1.959963984540054
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def predict(x, w, b) -> int:
    # Fixed label-0 tie break; exact ties yield 50% on a balanced schedule.
    return int(float((np.asarray(x) @ w.T + b).item()) > 1e-9)


def run_condition(c, sim_cfg, ports, args, name, out, *, network_factory=LIFNetwork):
    net = network_factory(c, sim_cfg, backend=args.backend, device="cpu" if args.backend == "torch" else None)
    if name == "disconnected":
        # Declared null: remove every anatomical edge before freezing the graph.
        net.silence({"index": np.arange(c.n)})
    lobe = None
    if name != "native":
        mode = name if name in {"leaky", "memoryless"} else "recurrent"
        conf = ProsthesisConfig(n_units=args.units, seed=args.seed, mode=mode,
                                tau_ms=args.tau_ms, max_drive_mV=args.max_drive_mV)
        lobe = net.attach_prosthesis(RecurrentProsthesis(c.n, net.dt, ports["read"], ports["write"], conf))
    handles = [net.inject_poisson(g, 0.0) for g in ports["cue"]]
    fixed_fly_weights = hashlib.sha256(net.w.tobytes()).hexdigest()

    def collect(labels, seeds):
        xs, hs, stats = [], [], []
        for label, seed in zip(labels, seeds):
            x, h, stat = trial(net, lobe, ports, handles, label, seed, args.cue_ms,
                               args.delay_ms, args.probe_ms, args.rate_hz)
            xs.append(x)
            if h is not None:
                hs.append(h)
            stats.append(stat)
        return np.array(xs), np.array(hs), stats

    calibration_labels, calibration_seeds = schedule(args.calibration, args.seed, 10)
    calibration_states = np.empty((0, args.units))
    calibration_mse = None
    if lobe is not None:
        lobe.feedback_enabled = False
        _, calibration_states, _ = collect(calibration_labels, calibration_seeds)
        targets = np.eye(2)[calibration_labels] * args.max_drive_mV
        calibration_mse = lobe.fit_feedback(calibration_states, targets, args.ridge)
        lobe.feedback_enabled = name != "read_only"
        lobe.save_weights(out / f"{name}_lobe.npz")

    train_labels, train_seeds = schedule(args.train, args.seed, 20)
    x_train, h_train, _ = collect(train_labels, train_seeds)
    y_train = (2 * train_labels - 1)[:, None]
    w_fly, b_fly = fit_ridge(x_train, y_train, args.ridge)
    w_lobe, b_lobe = (None, None) if lobe is None else fit_ridge(h_train, y_train, args.ridge)
    frozen_lobe = None if lobe is None else lobe.weights_digest()
    labels, seeds = schedule(args.evaluate, args.seed, 30)
    x_eval, h_eval, stats = collect(labels, seeds)
    predictions = [predict(x, w_fly, b_fly) for x in x_eval]
    lobe_predictions = [None] * len(labels) if lobe is None else [predict(h, w_lobe, b_lobe) for h in h_eval]
    assert fixed_fly_weights == hashlib.sha256(net.w.tobytes()).hexdigest(), "fly synapses changed"
    assert lobe is None or frozen_lobe == lobe.weights_digest(), "evaluation updated lobe weights"
    correct = int(np.sum(np.array(predictions) == labels))
    lobe_correct = None if lobe is None else int(np.sum(np.array(lobe_predictions) == labels))
    raw = [dict(trial=i, label=int(y), noise_seed=int(seed), prediction=p, lobe_prediction=hp,
                features=x.tolist(), **stat)
           for i, (y, seed, p, hp, x, stat) in enumerate(zip(labels, seeds, predictions, lobe_predictions, x_eval, stats))]
    (out / f"{name}_trials.json").write_text(json.dumps(raw, indent=2) + "\n")
    arrays = dict(calibration_labels=calibration_labels, calibration_seeds=calibration_seeds,
                  calibration_states=calibration_states, train_labels=train_labels,
                  train_seeds=train_seeds, train_fly_features=x_train, train_lobe_states=h_train,
                  eval_fly_features=x_eval, eval_lobe_states=h_eval, w_fly=w_fly, b_fly=b_fly)
    if lobe is not None:
        arrays.update(w_lobe=w_lobe, b_lobe=b_lobe)
    np.savez_compressed(out / f"{name}_data.npz", **arrays)
    return dict(condition=name, backend=net.backend, n_eval=len(labels), correct=correct,
                fly_accuracy=correct / len(labels), fly_wilson95=wilson(correct, len(labels)),
                lobe_only_accuracy=None if lobe_correct is None else lobe_correct / len(labels),
                lobe_only_wilson95=None if lobe_correct is None else wilson(lobe_correct, len(labels)),
                calibration_mse=calibration_mse, lobe_weights_sha256=frozen_lobe,
                fly_weights_sha256=fixed_fly_weights,
                lobe_config=None if lobe is None else asdict(lobe.config))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--toy", action="store_true")
    source.add_argument("--ports", type=Path, help="JSON target groups for the full cached FlyWire graph")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--out", type=Path, default=Path("results/prosthesis"))
    p.add_argument("--backend", choices=("numpy", "numba", "torch"), default="numpy")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--units", type=int, default=256)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--calibration", type=int, default=24)
    p.add_argument("--train", type=int, default=24)
    p.add_argument("--evaluate", type=int, default=40)
    p.add_argument("--cue-ms", type=float, default=100.0)
    p.add_argument("--delay-ms", type=float, default=500.0)
    p.add_argument("--probe-ms", type=float, default=100.0)
    p.add_argument("--tau-ms", type=float, default=500.0)
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--max-drive-mV", type=float, default=None,
                   help="default 12 for toy routing, 3 for full graph; bounds per target")
    p.add_argument("--ridge", type=float, default=1e-3)
    args = p.parse_args(argv)
    for count in (args.calibration, args.train, args.evaluate):
        if count < 2 or count % 2:
            p.error("calibration/train/evaluate must be even and >=2")
    if args.seed < 0 or args.units < 1:
        p.error("seed must be nonnegative and units must be positive")
    if args.backend == "numba" and not HAVE_NUMBA:
        p.error("numba is unavailable; explicitly choose --backend numpy")
    if args.backend == "torch" and not HAVE_TORCH:
        p.error("torch is unavailable; explicitly choose --backend numpy")
    if len(set(args.conditions)) != len(args.conditions):
        p.error("duplicate conditions")
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        p.error("ridge must be positive and finite")
    if args.toy:
        c, ports = make_toy()
        sim_cfg = dict(SIM)
    else:
        from .config import load_config
        from .data import load_connectome, cache_paths
        cfg = load_config(args.config)
        # Avoid implicit large downloads. Data preparation is a separate command.
        if not all(path.exists() for path in cache_paths(cfg)):
            p.error("FlyWire cache missing; first run python scripts/download_data.py")
        c = load_connectome(cfg)
        sim_cfg = dict(cfg["sim"])
        ports = json.loads(args.ports.read_text())
        # Background inputs, when enabled in this YAML, remain registered.
    ports = validate_ports(c, ports)
    args.max_drive_mV = args.max_drive_mV if args.max_drive_mV is not None else (12.0 if args.toy else 3.0)
    ProsthesisConfig(n_units=args.units, tau_ms=args.tau_ms, max_drive_mV=args.max_drive_mV, seed=args.seed)
    if not np.isfinite(args.rate_hz) or not 0 < args.rate_hz * sim_cfg["dt_ms"] * 1e-3 <= 1:
        p.error("rate-hz must be positive with rate*dt_seconds <= 1")
    for duration in (args.cue_ms, args.delay_ms, args.probe_ms):
        steps_for(duration, sim_cfg["dt_ms"])
        steps_for(duration, 10.0)  # align trial boundaries with complete lobe bins
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        p.error("output already contains summary.json; choose a new directory")
    start = time.monotonic()
    graph_hash = hashlib.sha256()
    for a in (c.root_ids, c.W.indptr, c.W.indices, c.W.data, c.sign):
        graph_hash.update(np.ascontiguousarray(a).tobytes())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(experiment="6-prototype", graph="synthetic-routing-64" if args.toy else "FlyWire-cache",
                  n_neurons=c.n, n_connected_pairs=c.W.nnz, graph_sha256=graph_hash.hexdigest(),
                  evidence_scope="software/toy only" if args.toy else "one modeled connectome, selected ports",
                  args=config, sim_config=sim_cfg, python=platform.python_version(), numpy=np.__version__,
                  ports_dense_indices={key: [g.tolist() for g in groups] for key, groups in ports.items()},
                  ports_root_ids={key: [c.root_ids[g].tolist() for g in groups] for key, groups in ports.items()},
                  source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("prosthesis.py", "prosthesis_memory.py", "sim.py")},
                  results=[], caveats=[
                      "Only prosthetic output weights and external scoring decoders learn; fly synapses are fixed.",
                      "Calibration observes delayed states without feedback; closed-loop distribution may differ.",
                      "Lobe-only accuracy measures external synthetic memory; it does not establish native memory.",
                      "Leaky control removes the recurrent matrix but retains slow state; memoryless removes both.",
                      "Disconnected control checks routing dependence, not an advantage of anatomical wiring.",
                      "No degree-preserving rewiring null, independent brains, or behavior/locomotion measured.",
                      "No advantage of 256 units or of recurrence is assumed; toy drive can force write-cell spikes.",
                      "Wilson intervals describe trial sampling within this run, not biological variability."])
    for name in args.conditions:
        result = run_condition(c, sim_cfg, ports, args, name, args.out)
        report["results"].append(result)
        print(f"{name}: fly={result['fly_accuracy']:.3f}, lobe={result['lobe_only_accuracy']}", flush=True)
        (args.out / "partial.json").write_text(json.dumps(report, indent=2) + "\n")
    report["wall_seconds"] = time.monotonic() - start
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "partial.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
