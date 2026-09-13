# FlyHamlet

The FlyWire fruit-fly connectome (v783, 139,255 neurons, 15.1 M connections, 54.5 M synapses)
run as a leaky integrate-and-fire network, dropped into a 2D arena that doubles as a
typewriter, and used for two experiments: how much entropy does the typing have, and can the
brain serve as an entropy source that eventually types Hamlet?

> **Where the randomness comes from.** The wiring is deterministic and the network is silent
> at rest: with no input and no noise, no neuron ever spikes and two runs are identical. Every
> bit of randomness in the keystroke logs and in the entropy-source experiment comes from the
> **injected Poisson background drive and the stochastic arena (looming) inputs**, seeded from
> `config.yaml`, not from the connectome itself. The brain shapes and mixes that noise; it does
> not create it.

## Layout

```
config.yaml              every gain, rate, path and seed
flyhamlet/data.py        Phase 0: download + cache the Codex v783 tables, build the Connectome
flyhamlet/sim.py         Phase 1: sparse LIF network (numba / numpy / torch backends), input + spike APIs
flyhamlet/validate.py    Phase 1: sugar GRN -> MN9 validation against Shiu et al. 2024
flyhamlet/arena.py       Phase 2: arena, DN motor readout, looming input
flyhamlet/typewriter.py  Phase 2: 3x9 key grid, keystroke log
flyhamlet/run_flies.py   Phase 2: N flies in parallel (optional --live matplotlib view, --tap)
flyhamlet/entropy_tap.py Phase 3: stream inter-spike intervals of a tapped neuron set to disk
analysis/entropy.py      Experiment 1: entropies, null models, Hamlet scoring, report + PNGs
analysis/whiten.py       Experiment 2: 64 ISIs -> SHA-256 -> rejection-sampled keys
analysis/typist.py       Experiment 2: longest Hamlet substring via a suffix automaton
analysis/entropy_tests.py Experiment 2: NIST SP 800-90B MCV + collision estimates, dieharder output
scripts/                 download_data.py, benchmark.py, benchmark_brian2.py, run_experiment{1,2}.sh
tests/                   unit tests (simulator vs Brian2, estimators, whitening, typist, tap)
docs/                    phase notes (DN screen)
results/                 validation, benchmark, arena logs, experiment reports
```

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py      # ~280 MB from the public flywire-data bucket, then builds data/cache
python -m pytest -q                  # 34 tests, ~15 s, no data download needed
```

## Phase 0: data

The public v783 release tables are served by Codex from
`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/` (the download page on
codex.flywire.ai needs a Google login; the bucket does not). Files used: `neurons.csv.gz`
(neurotransmitter predictions), `connections_no_threshold.csv.gz` (every synapse, no per-pair
threshold: this is exactly the 15,091,983-pair table Shiu et al. used for v783),
`classification_with_fw_and_hemibrain_types.csv.gz`, `consolidated_cell_types.csv.gz`,
`labels.csv.gz`, `names.csv.gz`.

`python -m flyhamlet.data` prints:

```
neurons: 139,255   connected pairs: 15,091,983   synapses: 54,492,922
nt: ACH=90,692 GLUT=24,854 GABA=19,176 SER=2,113 UNK=1,250 DA=935 OCT=235
sign: +1 = 95,225   -1 = 44,030
super_class: optic=77,536 central=32,388 sensory=16,903 visual_projection=8,053 ascending=2,362 descending=1,303 ...
```

Neurons without a prediction in `neurons.csv` (18,408) take the synapse-weighted majority
neurotransmitter of their output connections. Sign: ACh, DA, OCT, SER excitatory; GABA and
glutamate inhibitory (Shiu et al.). `Connectome.neurons_of_type`, `search_types` and
`search_labels` look names up in the annotation tables; nothing is addressed from memory.

## Phase 1: simulator

Shiu et al. 2024 (Nature, *A Drosophila computational brain model reveals sensorimotor
processing*) parameters, taken from their public Brian2 model (`philshiu/Drosophila_brain_model`):

| parameter | value |
|---|---|
| resting / reset / threshold | -52 / -52 / -45 mV |
| membrane / synaptic time constant | 20 / 5 ms |
| refractory period, synaptic delay | 2.2 ms, 1.8 ms |
| weight per synapse | 0.275 mV, connection weight = sign(pre) x synapse count x 0.275 mV |
| timestep | 0.1 ms (Brian2 default clock) |
| activation | Poisson events adding 68.75 mV to v (forces one spike per event), no refractory period on activated neurons |

The two linear ODEs are integrated exactly per step. Update order follows Brian2's schedule
(state update, threshold, synapses + Poisson, reset), and, as in Brian2, synaptic input that
arrives while the postsynaptic neuron is refractory is discarded. The implementation was
verified spike-for-spike against Brian2 2.9 on deterministic chains (`tests/test_sim.py`,
`tests/data/brian2_chain_*.json`); getting the refractory-input rule right moved the full-network
comparison from r = 0.95 to r = 0.99 (see below).

**Backends.** A custom sparse implementation: state updates are fused numba loops
(parallel over neurons) and spike delivery walks the CSR rows of the spiking neurons only;
the same algorithm exists in numpy and in PyTorch (for a GPU). All three produce bit-identical
spike trains on CPU. No GPU was available here, so the torch path is only timed on CPU.
`results/benchmark/benchmark.md` (4 CPU cores, 20 sugar GRNs driven at 150 Hz, 1 s simulated):

| backend | x real time |
|---|---|
| numba | ~3-4x |
| numpy | ~11x |
| torch (CPU) | ~40x |
| Brian2 (authors' model, cython) | ~70x |

**Determinism.** With no input and no background noise the network is at rest and no spike
ever occurs. All stochastic inputs draw from one `numpy.random.Generator(seed)`; same seed,
same inputs, same backend => identical spikes.

**APIs** (`flyhamlet/sim.py`): `inject_current(target, mV)`, `inject_poisson(target, rate_hz,
weight_mV, to='v'|'g')`, `set_background(rate_hz, weight_mV, target)`, `silence(target)`,
`subscribe(target, callback)` (per-timestep spike delivery), `step()`, `run()`. Targets are cell
type names, root-id lists, label regexes, super classes or index arrays.

**Validation** (`python -m flyhamlet.validate`, `results/validation/report.md`): the 20
labellar sugar GRNs of Shiu et al. that exist in v783 are driven at 10-200 Hz; the contralateral
MN9 fires from 25 Hz drive upward, MN6 and MN8 join at higher drive, the responder count is in
the low hundreds, and per-neuron rates match the authors' own Brian2 model run on the same
v783 tables with Pearson r = 0.987-0.998:

| GRN rate (Hz) | MN9 contra ours | MN9 contra Brian2 | responders ours / Brian2 | r |
|---|---|---|---|---|
| 50 | 10.2 | 14.5 | 289 / 276 | 0.987 |
| 100 | 62.8 | 62.0 | 374 / 366 | 0.994 |
| 150 | 79.2 | 88.5 | 393 / 385 | 0.996 |
| 200 | 90.2 | 90.0 | 430 / 415 | 0.998 |

## Phase 2: arena and typewriter

See `docs/phase2_dn_screen.md` for the annotation check and the looming screen that chose the
readout. Turning = `turn_gain` x (left - right) rate of DNa02 + DNa01 (both respond to
contralateral looming, so the fly turns away from walls); speed = `base_speed` + `speed_gain` x
DNp09 - `backward_gain` x MDN (moonwalker DNs drive backward walking). Walls within
`wall_distance_mm` drive that eye's LC4 + LPLC2 with proximity-scaled Poisson input. The arena
(90 x 30 mm) is a 3 x 9 grid of keys, letters assigned by a seeded permutation; a keystroke is
logged only when the fly enters a new key region (`sim_time_s,key_index,letter,x,y,heading_deg`).

```bash
python -m flyhamlet.run_flies --n-flies 4 --duration 300 --procs 4   # headless, seeds 42..45
python -m flyhamlet.run_flies --n-flies 1 --live                      # matplotlib live view
```

## Experiment 1: typing entropy

```bash
scripts/run_experiment1.sh          # or: python analysis/entropy.py --out results/entropy_exp1
```

Results (`results/entropy_exp1/report.md`; 4 flies x 300 s, seeds 42-45, 1,139 keystrokes,
0.95 keystrokes/s, no wall contact):

| bits per keystroke | fly typist | uniform grid walk (null) | uniform 27-key typist (null) |
|---|---|---|---|
| H0 marginal (Miller-Madow) | 4.70 | 4.73 | 4.75 |
| H(X_t+1 \| X_t) | 1.65 | 1.67 | 4.75 |
| LZ76 rate, length-matched | 1.89 | 2.45 | 4.16 |

All 27 keys are visited, the transition graph is one strongly connected component and has no
absorbing key. The marginal is nearly uniform but the conditional entropy is that of a random
walk on the grid: because a keystroke is only logged when the fly *enters* a region, consecutive
keys must be neighbours, and the LZ76 rate (1.89 bits) is below the grid walk's (2.45) because the
fly walks in long straight lines and reverses, which is more predictable than a random walk.

Hamlet (167,325 characters after cleaning) uses 468 distinct letter pairs; under this key layout
only 105 of them are grid-adjacent, so P(Hamlet) = 0 for *any* fly on this typewriter, not just
this one. The fitted model has 414 zero-probability pairs (150,475 of Hamlet's 167,324
transitions), listed with their Hamlet frequencies in the report. With Laplace smoothing the
expected number of keystrokes is ~10^313493, versus 27^len = 10^239503 for the uniform typist.

![transition heatmap](results/entropy_exp1/transition_heatmap.png)
![key visits](results/entropy_exp1/key_visit_heatmap.png)
![trajectory](results/entropy_exp1/trajectory.png)

## Experiment 2: fly brain as entropy source

```bash
scripts/run_experiment2.sh          # tapped flies -> whiten -> typist -> NIST tests + dieharder files
```

The tap samples 500 central-brain neurons (seeded; the list is dumped next to the ISI file)
that are neither of the wired sensory/motor types nor their direct synaptic partners, and
streams `(neuron, ISI in timesteps)` records to a binary file. `analysis/whiten.py` hashes
batches of 64 ISIs with SHA-256 and turns bytes into keys by rejection sampling (bytes >= 243
rejected, then mod 27); raw ISIs are never used mod 27. `analysis/typist.py` streams the keys
through a suffix automaton of Hamlet and reports the longest Hamlet substring ever produced.
`analysis/entropy_tests.py` gives the NIST SP 800-90B most-common-value and collision
min-entropy estimates on the raw ISIs and writes raw and whitened 32-bit streams for
`dieharder -g 201 -f <file>`.

Results (`results/entropy_exp2/report.md`; 4 flies x 120 s, background 200 Hz x 4 mV, 500 tapped
neurons per fly of which ~430 fire, median 2.6 Hz): 3,603,323 raw ISIs -> 56,301 SHA-256 batches
-> 1,709,435 keys (accept fraction 0.949).

| data | NIST 800-90B estimator | min-entropy |
|---|---|---|
| raw ISI symbols (26,477 distinct values) | most common value | 7.55 bits / sample |
| raw ISI bits (21-bit words) | MCV / collision | 0.34 / 0.20 bits / bit |
| raw ISI low byte | MCV | 7.15 bits / byte |
| whitened SHA-256 bytes | MCV | 7.91 bits / byte |
| whitened SHA-256 bits | collision | 0.94 bits / bit |

The raw ISIs carry real entropy (about 7.5 bits per interval by the MCV bound) but are far from
uniform as words: the high bits are almost always zero, which is exactly why they are hashed
rather than used mod 27. Typist: after 1.7 M keys the longest substring of Hamlet produced is
8 characters (`'c look y'`, found at key 26,236) at 2.3 M keys/s through the suffix automaton;
a 9-character record is expected after ~3e7 keys and Hamlet itself after ~27^167325 = 10^239503.

dieharder (`results/entropy_exp2/dieharder.txt`, subset sts_monobit / sts_runs / sts_serial /
diehard_runs / operm5 / rank32x32, `-g 201` raw 32-bit words): the raw ISI stream fails 34 of 36
results outright (2 weak); the whitened stream gives 11 FAILED / 11 WEAK / 14 PASSED. That
whitened profile is not a defect of the source: a same-sized (1.8 MB) `/dev/urandom` control file
run through the identical subset gives 9 FAILED / 13 WEAK / 12 PASSED, because dieharder needs
gigabytes and rewinds a small file many times per test (all 56,301 SHA-256 digests in the
whitened stream are distinct, byte mean 127.54). A full-battery dieharder run needs a much longer
tap (`entropy_tap.duration_s`) and `scripts/run_dieharder.sh`.

## Notes and caveats

* The FlyWire volume is a brain without a ventral nerve cord: descending-neuron activity is
  read out directly as locomotion with hand-set gains. The gains, the base speed and the
  looming gain are the free parameters of Phase 2 and are all in `config.yaml`.
* Shiu et al. note that absolute firing rates of the model are not to be trusted; the
  validation is about which neurons respond and how the response scales.
* 616 neurons in `neurons.csv` have no connections at all and never do anything.

## References

* Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult brain*, Nature; Schlegel et al.
  2024, *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (FlyWire v783,
  https://codex.flywire.ai).
* Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*,
  Nature; code at https://github.com/philshiu/Drosophila_brain_model.
* Hamlet: Project Gutenberg #1524.
