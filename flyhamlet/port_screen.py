"""Select disjoint neural ports on development trials, then validate once on fresh seeds."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .activity import ActivityAbort, ActivityLimits, GuardedNetwork
from .prosthesis import ProsthesisConfig, RecurrentProsthesis
from .prosthesis_memory import SIM, make_toy, steps_for, validate_ports
from .sim import HAVE_NUMBA, HAVE_TORCH, resolve_target


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def graph_hash(c):
    h = hashlib.sha256()
    for a in (c.root_ids, c.W.indptr, c.W.indices, c.W.data, c.sign):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def source_hashes():
    names = ("sim.py", "data.py", "prosthesis.py", "prosthesis_memory.py",
             "activity.py", "port_screen.py", "screened_memory.py")
    return {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"output must be empty: {path}")
    return path


def load_source(toy, config):
    if toy:
        c, ports = make_toy()
        # A separate, explicitly synthetic fixture: make write -> read suprathreshold.
        # The original toy's weak feedback path cannot pass this spike-effect screen.
        w = c.W.tolil()
        for side in range(2):
            for i in ports["write"][side]:
                for j in ports["read"][side]:
                    w[i, j] = 40
        c.W = w.tocsr()
        specs = dict(cue_pairs=[[{"index": g.tolist()} for g in ports["cue"]]],
                     **{name + "_pool": {"index": np.concatenate(ports[name]).tolist()}
                        for name in ("read", "write", "report")})
        return c, dict(SIM), specs
    from .config import load_config
    from .data import cache_paths, load_connectome
    cfg = load_config(config)
    if not all(p.exists() for p in cache_paths(cfg)):
        raise ValueError("FlyWire cache missing; run python scripts/download_data.py first")
    return load_connectome(cfg), dict(cfg["sim"]), None


@dataclass(frozen=True)
class ScreenConfig:
    seed: int = 41
    group_size: int = 8
    development: int = 4           # trials per side; common noise for paired interventions
    validation: int = 8
    cue_ms: float = 100.0
    write_ms: float = 200.0
    rate_hz: float = 100.0
    min_effect_hz: float = 2.0
    min_success: float = 0.75
    drives_mV: tuple = (3.0, 6.0, 9.0, 12.0)

    def __post_init__(self):
        for name in ("seed", "group_size", "development", "validation"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else (2 if name in ("development", "validation") else 1)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid {name}")
        for name in ("cue_ms", "write_ms", "rate_hz", "min_effect_hz", "min_success"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        if self.min_success > 1 or not self.drives_mV:
            raise ValueError("invalid success threshold or empty drive sweep")
        if any(not np.isfinite(v) or v <= 0 for v in self.drives_mV):
            raise ValueError("drive sweep must be positive and finite")
        if list(self.drives_mV) != sorted(set(self.drives_mV)):
            raise ValueError("drive sweep must be strictly increasing")


class ScreenRejected(ValueError):
    pass


def select_opponent_groups(delta, pool, excluded, size, minimum):
    """Stable ranking of opposite effects; each group has at most size cells."""
    pool = np.setdiff1d(pool, excluded)
    result = []
    for direction in (1, -1):
        score = direction * delta[pool]
        ids = pool[score >= minimum]
        values = direction * delta[ids]
        ids = ids[np.lexsort((ids, -values))][:size]
        if not len(ids):
            raise ScreenRejected("no distinguishable opposing populations in candidate pool")
        result.append(np.sort(ids))
    return result


def select_writes(c, read, pool, report_pool, excluded, size):
    """Rank excitatory cells with direct anatomical routes to read and report pools."""
    to_report = np.asarray(c.W[:, report_pool].sum(axis=1)).ravel()
    to_read = [np.asarray(c.W[:, g].sum(axis=1)).ravel() for g in read]
    score = (to_read[0] - to_read[1]) * np.log1p(to_report)
    eligible = pool[(c.sign[pool] > 0) & (to_report[pool] > 0)]
    return select_opponent_groups(score, eligible, excluded, size, 1e-12)


def port_metrics(cue_rates, write_rates, off_rates, ports, cfg):
    result = {}
    for side in range(2):
        other = 1 - side
        read, report = ports["read"], ports["report"]
        cue_margin = cue_rates[side][:, read[side]].mean(1) - cue_rates[side][:, read[other]].mean(1)
        report_margin = write_rates[side][:, report[side]].mean(1) - write_rates[side][:, report[other]].mean(1)
        for name, margins in (("cue_read", cue_margin), ("write_report_selectivity", report_margin)):
            result[f"{name}_{side}"] = dict(success=float(np.mean(margins >= cfg.min_effect_hz)),
                                            effects_hz=margins.tolist())
        for name in ("read", "report"):
            group = ports[name][side]
            effect = (write_rates[side][:, group] - off_rates[:, group]).mean(1)
            result[f"write_{name}_effect_{side}"] = dict(
                success=float(np.mean(effect >= cfg.min_effect_hz)), effects_hz=effect.tolist())
    result["passed"] = all(v["success"] >= cfg.min_success for v in result.values())
    return result


def run_screen(c, sim_cfg, specs, cfg, limits, backend, out, *, toy=False):
    """Freeze the first development-passing candidate, validate once, fail closed.

    No memory-benchmark trials enter selection. Validation failure ends this run;
    it does not trigger a search through further candidates or amplitudes.
    """
    out = fresh_directory(out)
    if set(specs) != {"cue_pairs", "read_pool", "write_pool", "report_pool"} or not specs["cue_pairs"]:
        raise ValueError("candidates need cue_pairs, read_pool, write_pool, report_pool")
    for duration in (cfg.cue_ms, cfg.write_ms):
        steps_for(duration, sim_cfg["dt_ms"])
        steps_for(duration, 10.0)
    if cfg.rate_hz * sim_cfg["dt_ms"] * 1e-3 > 1:
        raise ValueError("cue rate * timestep must be <= 1")
    pools = {k: resolve_target(c, specs[k + "_pool"]) for k in ("read", "write", "report")}
    for name, pool in pools.items():
        if not len(pool) or np.any(pool < 0) or np.any(pool >= c.n):
            raise ValueError(f"invalid or empty {name} pool")
    report = dict(schema=1, status="running", toy=bool(toy), graph_sha256=graph_hash(c),
                  n_neurons=c.n, sim_sha256=json_hash(sim_cfg), sim_config=sim_cfg,
                  screen_config=asdict(cfg), limits=asdict(limits), backend=backend,
                  candidates=specs, candidates_sha256=json_hash(specs), source_sha256=source_hashes(),
                  python=platform.python_version(), numpy=np.__version__, attempts=[],
                  development_noise_seeds=[], validation_noise_seeds=[])
    records = []
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 606]))
    # Generate all noise seeds before any outcomes, with explicit uniqueness.
    seed_count = len(specs["cue_pairs"]) * cfg.development + cfg.validation
    seeds = rng.choice(2**32, size=seed_count, replace=False).tolist()
    report["development_noise_seeds"] = seeds[:-cfg.validation]
    report["validation_noise_seeds"] = seeds[-cfg.validation:]
    write_json(out / "screen.json", report)
    selected = None
    net = None
    try:
        for pair_index, cue_specs in enumerate(specs["cue_pairs"]):
            attempt = dict(candidate=pair_index, amplitudes=[])
            report["attempts"].append(attempt)
            try:
                cue = [resolve_target(c, s) for s in cue_specs]
                if len(cue) != 2 or any(not len(g) for g in cue):
                    raise ScreenRejected("cue pair must contain two nonempty groups")
                excluded_cue = np.concatenate(cue)
                if len(np.unique(excluded_cue)) != len(excluded_cue):
                    raise ScreenRejected("cue groups overlap")
                if np.any(excluded_cue < 0) or np.any(excluded_cue >= c.n):
                    raise ScreenRejected("cue indices outside graph")
            except (KeyError, ScreenRejected) as e:
                attempt["rejected"] = str(e)
                continue
            net = GuardedNetwork(c, sim_cfg, backend=backend, device="cpu", limits=limits,
                                 activity_records=records, groups={f"cue{i}": g for i, g in enumerate(cue)})
            handles = [net.inject_poisson(g, 0.0) for g in cue]
            lobe = None

            def probe(mode, side, noise_seed, drive=0.0):
                net.reset_state(seed=int(noise_seed))
                for i, handle in enumerate(handles):
                    net.set_poisson_rate(handle, cfg.rate_hz if mode == "cue" and side == i else 0.0)
                if lobe is not None:
                    lobe.feedback_enabled = mode == "write"
                    lobe.bias.fill(0)
                    if mode == "write":
                        lobe.bias[side] = drive
                ms = cfg.cue_ms if mode == "cue" else cfg.write_ms
                counts = np.zeros(c.n, dtype=np.int64)
                for _ in range(steps_for(ms, net.dt)):
                    np.add.at(counts, net.step(), 1)
                net.finish_trial()
                return counts / (ms * 1e-3)

            def collect(noise_seeds, drive=None):
                cue_rates = np.array([[probe("cue", side, s) for s in noise_seeds] for side in (0, 1)])
                if drive is None:
                    return cue_rates
                off = np.array([probe("off", 0, s) for s in noise_seeds])
                writes = np.array([[probe("write", side, s, drive) for s in noise_seeds] for side in (0, 1)])
                return cue_rates, writes, off

            development_seeds = seeds[pair_index * cfg.development:(pair_index + 1) * cfg.development]
            cue_rates = collect(development_seeds)
            try:
                excluded = np.union1d(excluded_cue, pools["report"])
                read = select_opponent_groups(cue_rates[0].mean(0) - cue_rates[1].mean(0),
                                              pools["read"], excluded, cfg.group_size, cfg.min_effect_hz)
                excluded = np.union1d(excluded, np.concatenate(read))
                write = select_writes(c, read, pools["write"], pools["report"], excluded, cfg.group_size)
            except ScreenRejected as e:
                attempt["rejected"] = str(e)
                continue
            lobe = net.attach_prosthesis(RecurrentProsthesis(c.n, net.dt, read, write,
                    ProsthesisConfig(n_units=1, max_drive_mV=max(cfg.drives_mV), seed=cfg.seed)))
            net.activity.set_groups({f"{name}{i}": g for name, gs in
                                     dict(cue=cue, read=read, write=write).items() for i, g in enumerate(gs)})
            off = np.array([probe("off", 0, s) for s in development_seeds])
            for drive in cfg.drives_mV:
                writes = np.array([[probe("write", side, s, drive) for s in development_seeds] for side in (0, 1)])
                amp = dict(drive_mV=drive)
                attempt["amplitudes"].append(amp)
                reserved = np.concatenate(cue + read + write)
                try:
                    reporters = select_opponent_groups(writes[0].mean(0) - writes[1].mean(0),
                                    pools["report"], reserved, cfg.group_size, cfg.min_effect_hz)
                except ScreenRejected as e:
                    amp["rejected"] = str(e)
                    continue
                ports = validate_ports(c, dict(cue=cue, read=read, write=write, report=reporters))
                amp["metrics"] = port_metrics(cue_rates, writes, off, ports, cfg)
                if amp["metrics"]["passed"]:
                    selected = ports
                    report["selected_candidate"] = pair_index
                    report["drive_mV"] = drive
                    break
            write_json(out / "screen.json", report)
            if selected is not None:
                break
        if selected is None:
            raise ScreenRejected("no development candidate passed; no ports exported")
        # No further candidate selection after this point.
        net.activity.set_groups({f"{name}{i}": g for name, gs in selected.items() for i, g in enumerate(gs)})
        validation = collect(report["validation_noise_seeds"], report["drive_mV"])
        metrics = port_metrics(*validation, selected, cfg)
        report["validation"] = metrics
        if not metrics["passed"]:
            raise ScreenRejected("frozen candidate failed fresh-seed validation; no ports exported")
        serial = {k: [{"ids": c.root_ids[g].tolist()} for g in gs] for k, gs in selected.items()}
        report.update(status="validated", ports_sha256=json_hash(serial),
                      ports=serial, evidence_scope="synthetic software check" if toy else
                      "development validation on one simulated connectome; no delayed-memory result")
        write_json(out / "ports.json", serial)
    except (ActivityAbort, FloatingPointError, ScreenRejected) as e:
        report.update(status="aborted" if isinstance(e, (ActivityAbort, FloatingPointError)) else "rejected",
                      failure=e.details if isinstance(e, ActivityAbort) else {"reason": str(e)})
        if net is not None and net.activity.aborted is not None:
            report["failed_trial_windows"] = list(net.activity.windows)
        raise
    finally:
        write_json(out / "activity.json", records)
        write_json(out / "screen.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--candidates", type=Path, default=Path("configs/prosthesis_screen.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("numpy", "numba", "torch"), default="numba")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--development", type=int, default=4)
    p.add_argument("--validation", type=int, default=8)
    p.add_argument("--limits", type=Path, help="JSON ActivityLimits overrides")
    args = p.parse_args(argv)
    try:
        if args.backend == "numba" and not HAVE_NUMBA or args.backend == "torch" and not HAVE_TORCH:
            raise ValueError("requested backend is unavailable")
        c, sim_cfg, toy_specs = load_source(args.toy, args.config)
        specs = toy_specs if args.toy else json.loads(args.candidates.read_text())
        limits = ActivityLimits(**json.loads(args.limits.read_text())) if args.limits else ActivityLimits()
        cfg = ScreenConfig(seed=args.seed, development=args.development, validation=args.validation)
        report = run_screen(c, sim_cfg, specs, cfg, limits, args.backend, args.out, toy=args.toy)
        print(f"{report['status']}: {args.out / 'ports.json'}; drive={report['drive_mV']} mV")
        return 0
    except (ValueError, ActivityAbort, FloatingPointError) as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
