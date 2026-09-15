"""Run a guarded delayed-cue benchmark using a frozen, validated port manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .activity import ActivityAbort, ActivityLimits, GuardedNetwork
from .port_screen import (fresh_directory, graph_hash, json_hash, load_source,
                          source_hashes, write_json)
from .prosthesis_memory import CONDITIONS, run_condition, schedule, steps_for, validate_ports
from .sim import HAVE_NUMBA, HAVE_TORCH


def validate_manifest(manifest, serial_ports, c, sim_cfg, *, toy, schedules):
    """Fail on stale/edited ports, source, graph, simulation, or reused noise seeds."""
    if manifest.get("schema") != 1 or manifest.get("status") != "validated":
        raise ValueError("screen manifest did not pass validation")
    if manifest.get("toy") is not bool(toy):
        raise ValueError("synthetic and full-connectome evidence cannot be interchanged")
    expected = dict(graph_sha256=graph_hash(c), sim_sha256=json_hash(sim_cfg),
                    ports_sha256=json_hash(serial_ports), source_sha256=source_hashes())
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"screen manifest mismatch: {name}; repeat screening")
    validation = manifest.get("validation", {})
    if validation.get("passed") is not True:
        raise ValueError("screen manifest lacks passed validation metrics")
    used = set(manifest["development_noise_seeds"] + manifest["validation_noise_seeds"])
    for count, seed, phase in schedules:
        noise = set(map(int, schedule(count, seed, phase)[1]))
        if noise & used:
            raise ValueError("benchmark noise seeds overlap port-screen development/validation")
    return validate_ports(c, serial_ports)


def run_benchmark(args):
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("benchmark seeds must be unique and nonnegative")
    if not args.seeds or args.units < 1:
        raise ValueError("need benchmark seeds and positive unit count")
    if args.backend == "numba" and not HAVE_NUMBA or args.backend == "torch" and not HAVE_TORCH:
        raise ValueError("requested backend is unavailable")
    if not args.conditions or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("conditions must be nonempty and unique")
    if any(name not in CONDITIONS for name in args.conditions):
        raise ValueError("unknown condition")
    manifest = json.loads(args.manifest.read_text())
    serial = json.loads(args.ports.read_text())
    c, sim_cfg, _ = load_source(args.toy, args.config)
    schedules = [(n, seed, phase) for seed in args.seeds for n, phase in
                 ((args.calibration, 10), (args.train, 20), (args.evaluate, 30))]
    ports = validate_manifest(manifest, serial, c, sim_cfg, toy=args.toy, schedules=schedules)
    if manifest["backend"] != args.backend:
        raise ValueError("benchmark backend must match the screened backend")
    limits = ActivityLimits(**manifest["limits"])
    for ms in (args.delay_ms, 100.0, manifest["screen_config"]["cue_ms"]):
        steps_for(ms, sim_cfg["dt_ms"])
        steps_for(ms, 10.0)
    out = fresh_directory(args.out)
    report = dict(schema=1, status="running", toy=args.toy, graph_sha256=graph_hash(c),
                  ports_sha256=json_hash(serial), source_sha256=source_hashes(),
                  manifest_sha256=json_hash(manifest), screen_manifest=manifest,
                  python=platform.python_version(), numpy=np.__version__, sim_config=sim_cfg,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  limits=asdict(limits), results=[],
                  caveats=["One modeled graph; no animal or biological-variability evidence.",
                           "Screen validation tests routing, not delayed recall.",
                           "Feedback stays active throughout the delay and scoring interval.",
                           "Guard limits are engineering stop rules, not physiological safety bounds.",
                           "Seed-wise accuracy differences do not establish an anatomical wiring advantage."])
    write_json(out / "status.json", report)
    start = time.monotonic()
    current = None
    try:
        for seed in args.seeds:
            seed_dir = out / f"seed_{seed}"
            seed_dir.mkdir()
            settings = argparse.Namespace(seed=seed, backend=args.backend, units=args.units,
                    calibration=args.calibration, train=args.train, evaluate=args.evaluate,
                    cue_ms=manifest["screen_config"]["cue_ms"], delay_ms=args.delay_ms,
                    probe_ms=100.0, tau_ms=500.0, rate_hz=manifest["screen_config"]["rate_hz"],
                    max_drive_mV=manifest["drive_mV"], ridge=1e-3)
            for condition in args.conditions:
                current = dict(seed=seed, condition=condition)
                records, networks = [], []

                def factory(*a, **kw):
                    net = GuardedNetwork(*a, **kw, limits=limits, activity_records=records,
                            groups={f"{k}{i}": g for k, gs in ports.items() for i, g in enumerate(gs)})
                    networks.append(net)
                    return net

                try:
                    result = run_condition(c, sim_cfg, ports, settings, condition, seed_dir,
                                           network_factory=factory)
                    for net in networks:
                        net.finish_trial()
                finally:
                    write_json(seed_dir / f"{condition}_activity.json", records)
                result["seed"] = seed
                report["results"].append(result)
                write_json(out / "status.json", report)
                print(f"seed={seed} {condition}: {result['correct']}/{result['n_eval']}", flush=True)
        differences = []
        for seed in args.seeds:
            scores = {r["condition"]: r["fly_accuracy"] for r in report["results"] if r["seed"] == seed}
            for control in ("native", "read_only", "memoryless", "leaky", "disconnected"):
                if "closed_loop" in scores and control in scores:
                    differences.append(dict(seed=seed, control=control,
                                             accuracy_difference=scores["closed_loop"] - scores[control]))
        report.update(status="completed", wall_seconds=time.monotonic() - start,
                      paired_seed_differences=differences)
        write_json(out / "summary.json", report)
    except (ActivityAbort, FloatingPointError) as e:
        report.update(status="aborted", failed_run=current,
                      failure=e.details if isinstance(e, ActivityAbort) else {"reason": str(e)})
        raise
    except Exception as e:
        report.update(status="failed", failed_run=current, failure={"reason": str(e)})
        raise
    finally:
        write_json(out / "status.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--ports", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("numpy", "numba", "torch"), default="numba")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--units", type=int, default=256)
    p.add_argument("--delay-ms", type=float, default=500.0)
    p.add_argument("--calibration", type=int, default=24)
    p.add_argument("--train", type=int, default=24)
    p.add_argument("--evaluate", type=int, default=100)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = p.parse_args(argv)
    try:
        run_benchmark(args)
        return 0
    except (ValueError, ActivityAbort, FloatingPointError) as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
