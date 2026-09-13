"""EntropyTap: ISI recording and binary round-trip on the toy network."""
import numpy as np
from pathlib import Path

from flyhamlet.entropy_tap import EntropyTap, read_isi_file
from flyhamlet.sim import LIFNetwork
from test_sim import toy, SIM


def test_tap_records_isis(tmp_path: Path):
    c = toy()
    net = LIFNetwork(c, SIM, seed=5, backend="numpy")
    tap = EntropyTap(indices=np.array([1, 2]), root_ids=c.root_ids[[1, 2]], out_path=tmp_path / "isi.bin", dt_ms=SIM["dt_ms"])
    tap.attach(net)
    net.inject_poisson({"index": [1]}, rate_hz=2000.0)
    rec = net.run(duration_s=0.2, callback=tap.on_step)
    tap.close()
    slots, isis, n, dt = read_isi_file(tmp_path / "isi.bin")
    assert n == 2 and dt == SIM["dt_ms"]
    for slot, neuron in ((0, 1), (1, 2)):
        st = rec.steps[rec.neurons == neuron]
        assert isis[slots == slot].tolist() == np.diff(st).tolist()
    assert len(isis) == tap.n_isi > 50
