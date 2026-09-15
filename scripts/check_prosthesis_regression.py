"""Compare the patched simulator with its hash-verified, unmodified baseline."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flyhamlet.sim import LIFNetwork, HAVE_NUMBA, HAVE_TORCH
from flyhamlet.prosthesis_memory import make_toy, SIM

BASE_BLOB = "9428cf8655b311c5c1e45a72c9803c5eabad88f1"


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_state(net):
    if net.backend == "torch":
        net._sync_from_torch()
    return net.v.copy(), net.g.copy(), net.rfc_left.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "flyhamlet/sim.py.prosthesis-backup")
    parser.add_argument("--out", type=Path, default=ROOT / "results/prosthesis_regression.json")
    args = parser.parse_args()
    data = args.baseline.read_bytes()
    if blob(data) != BASE_BLOB:
        parser.error("baseline hash differs from the inspected upstream simulator")
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline_sim.py"
        path.write_bytes(data)
        name = "flyhamlet._prosthesis_baseline"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        c, ports = make_toy()
        backends = ["numpy"] + (["numba"] if HAVE_NUMBA else []) + (["torch"] if HAVE_TORCH else [])
        for backend in backends:
            for seed in (6, 11, 42):
                a = module.LIFNetwork(c, SIM, seed=seed, backend=backend, device="cpu")
                b = LIFNetwork(c, SIM, seed=seed, backend=backend, device="cpu")
                for net in (a, b):
                    net.inject_poisson(ports["cue"][0], 200)
                    net.inject_current(ports["write"][0], 1)
                    net.set_background(50, .75)
                for reset in (False, True):
                    if reset:
                        for net in (a, b):
                            net.reset_state(seed=seed + 1000)
                    ra, rb = a.run(duration_s=.08), b.run(duration_s=.08)
                    np.testing.assert_array_equal(ra.steps, rb.steps)
                    np.testing.assert_array_equal(ra.neurons, rb.neurons)
                    for x, y in zip(read_state(a), read_state(b)):
                        np.testing.assert_array_equal(x, y)
                    assert a.rng.bit_generator.state == b.rng.bit_generator.state
                checks.append(dict(backend=backend, seed=seed, spikes_and_state_exact=True,
                                   rng_exact=True, reset_checked=True))
    report = dict(baseline_git_blob=BASE_BLOB, baseline_scope="exact sim.py, 64-cell synthetic graph",
                  patched_sha256=hashlib.sha256((ROOT / "flyhamlet/sim.py").read_bytes()).hexdigest(),
                  full_repository_suite_run=False, checks=checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(checks)} exact baseline comparisons passed (including reset/replay)")


if __name__ == "__main__":
    main()
