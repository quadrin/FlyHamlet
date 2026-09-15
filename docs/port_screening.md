# Screened ports and guarded delayed-cue experiments

This workflow selects cue/read/write/report populations, checks their routing on
fresh development-validation trials, and freezes the resulting ports before a
separate delayed-memory benchmark. It extends the Python prototype. Native
simulation dynamics, the browser demo, configuration defaults, and previously
published results remain unchanged.

## Evidence scope

Full FlyWire screening and full-connectome delayed-memory experiments have not
been executed for this change. `configs/prosthesis_screen.json` contains candidate
populations, not validated anatomical ports. Only a successful screen writes
`ports.json`; an unsuccessful screen exits with code 2 and writes diagnostic
`screen.json` without usable ports. A validated screen establishes the declared
routing criteria on development trials. It supplies no delayed-memory result.

The synthetic fixture used here differs from the original memory toy: its
write-to-read connections have 40 synapses per pair instead of one. This makes
spike-mediated feedback measurable. The original toy fails the new feedback
criterion. Neither fixture supplies evidence about the actual fly connectome.

## Full-cache workflow

From a checkout with the repository dependencies installed:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m pytest -q
node --test tests/*.cjs

# Explicit download/cache preparation; omit when the cache already exists.
python scripts/download_data.py

python -m flyhamlet.port_screen \
  --config config.yaml --candidates configs/prosthesis_screen.json \
  --backend numba --seed 41 --development 4 --validation 8 \
  --out results/prosthesis_screen_v1

# Run only after screen.json reports status: validated.
python -m flyhamlet.screened_memory \
  --config config.yaml --backend numba \
  --ports results/prosthesis_screen_v1/ports.json \
  --manifest results/prosthesis_screen_v1/screen.json \
  --seeds 101 102 103 --units 256 --delay-ms 500 \
  --calibration 24 --train 24 --evaluate 100 \
  --out results/prosthesis_memory_v1
```

Both commands require an empty output directory and refuse overwrites. Screening
requires an existing FlyWire cache and performs no implicit large download.
The seed counts are development/evaluation samples from one fixed modeled graph;
they do not represent independent animals. Repeat screening after changing the
simulation or any source file covered by the manifest. Use the same backend for
screening and benchmarking; Torch runs on CPU in this workflow.

## Selection protocol

The supplied candidates try bilateral LC4 cues, followed by bilateral LPLC2 cues.
Read/write pools contain central neurons; report candidates contain descending
neurons. These are hypotheses to test. Missing types, absent routes, insufficient
responses, and guard trips can prevent export.

For each candidate pair, four development trials per side deliver 100 Hz Poisson
cues for 100 ms. The screen ranks opposing cue responses and selects up to eight
read neurons per channel, excluding cue and reserved report neurons. It ranks
excitatory write candidates with direct anatomical connections to the selected
read channels and the report pool. This deliberately narrow search can miss
usable multi-hop or inhibitory interfaces; failure is not evidence that no such
interface exists.

The declared write-drive sweep is 3, 6, 9, then 12 mV. Each intervention lasts
200 ms and uses the prosthesis's actual bounded synaptic-slot delivery. The
screen compares each write channel against the other channel and a paired
zero-drive trial. It selects opposing report responses on development data.
All four ports remain disjoint. Candidate ranking never uses memory-test trials.

The first candidate/amplitude that passes development freezes the ports. Eight
fresh noise seeds per side then validate it once. Every side must achieve at
least 75% success for all four criteria: cue-to-read selectivity,
write-to-report selectivity, write-to-read excitation versus zero drive, and
write-to-report excitation versus zero drive. Each successful trial must exceed
or equal a 2 Hz mean-rate effect. A validation failure rejects the whole run;
it does not search another candidate using those validation outcomes.

These thresholds are predeclared engineering checks, not significance tests or
proof of generalization. Amplitude selection favors the smallest declared drive
that passes development. A 9 or 12 mV command can directly make write cells fire;
that fact alone does not show an advantage from anatomical wiring. The screen
records every attempted candidate, drive, rate effect, seed, graph hash, root-ID
port, simulation setting, source hash, and activity limit.

## Activity monitoring and failure handling

Every screening and benchmark trial uses `GuardedNetwork`, an opt-in wrapper
around the unchanged simulator. It counts all spikes and checks nonoverlapping
50 ms windows, plus any final partial window. Default stop limits are a 100 Hz
whole-network mean, 500 Hz for any neuron, and 250 Hz for monitored populations.
It also stops on nonfinite membrane or synaptic state at those boundaries.
It monitors calibration, decoder training, cue, blank-delay, and scoring phases.
The guard latches after a violation and prevents further stepping until an
explicit reset; the command aborts the run and never auto-resumes it.

These thresholds bound an experimental operating range. They are not animal
safety limits or validated physiological ranges. Shorter bursts can occur inside
a window; the guard does not constrain sub-window instantaneous rates. Exported
logs identify each window and trial noise seed. Supply a JSON file with
`ActivityLimits` fields through the screen's `--limits` option to predeclare
other limits. The guarded benchmark inherits the recorded limits.

A benchmark guard trip leaves `status.json` marked `aborted`, identifies the
failed condition/seed/window, and creates no successful `summary.json`.
Completed earlier conditions and activity logs remain available. Generic run
errors leave status `failed`; an interrupted process can leave status `running`,
which must not be treated as a completed result.

## Benchmark outputs and controls

The gated benchmark verifies graph, simulator settings, source hashes, and the
root-ID port file against the validated screen manifest. It rejects reused
screening noise seeds. This detects stale or accidentally edited inputs; the
manifest is an audit record, not a cryptographic attestation of an experiment.
The older `prosthesis_memory` command remains available as an ungated low-level
runner. Use `screened_memory` for the workflow described here.

All six existing conditions run by default: native, read-only, memoryless,
leaky, closed-loop, and disconnected. The benchmark preserves separate
calibration, decoder-training, and held-out schedules, resets dynamics between
trials, and freezes trained weights for evaluation. It records lobe-only and
downstream decoding scores, per-trial results, activity windows, and seed-wise
closed-loop accuracy differences against each control. Confidence intervals in
the underlying condition results describe within-run trial sampling only.
Feedback stays active during the blank interval and scoring period; this does
not test persistence after prosthesis removal. Anatomy-rewiring controls,
behavioral navigation, GPU validation, and minimum-unit sweeps remain further
experiments.

## No-download software check

```bash
python -m flyhamlet.port_screen --toy --backend numpy \
  --development 2 --validation 2 --out results/screen_toy
python -m flyhamlet.screened_memory --toy --backend numpy \
  --ports results/screen_toy/ports.json --manifest results/screen_toy/screen.json \
  --seeds 101 --units 32 --calibration 4 --train 4 --evaluate 8 \
  --out results/memory_toy
```

Local validation for this change ran 91 tests (46 existing prosthesis tests and
45 new guard/screen/benchmark tests) on the exact merged simulator and data
loader sources, covering NumPy, Numba, and Torch CPU. The synthetic smoke run
selected 9 mV after rejecting 3 and 6 mV; its recurrent and leaky feedback
conditions each scored 8/8, while the other four conditions scored 4/8. This
small engineered example establishes no measured recurrence advantage.
The added CI workflow runs all repository Python tests, browser `.cjs` tests,
and the exact pre-prosthesis simulator regression. Its outcome must be checked
on the pull request; local subset tests do not establish a full-suite pass.
