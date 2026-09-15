# Experiment 6 prototype: synthetic memory interface

Scope: the implementation augments the Python LIF simulation. Artificial units maintain continuous-valued state outside the anatomical adjacency matrix and participate in the simulation through an explicit feedback interface. The browser worker has no corresponding port. No living animal was modified, and no full FlyWire experiment was executed for this package.

Run from a repository checkout

The interface is opt-in; existing simulations and the browser demo remain unchanged.
With the repository dependencies and pytest installed:

```bash
python -m pytest -q tests/test_prosthesis.py
python -m flyhamlet.prosthesis_memory --toy --backend numpy --out results/prosthesis
```

The toy run needs no FlyWire download. Inspect all runner options with
`python -m flyhamlet.prosthesis_memory --help`. A full-connectome run requires
`--config config.yaml --ports ports.json`; supply disjoint cue/read/write/report
populations through that JSON file. Full-connectome results remain untested.

To compare against the exact pre-prosthesis simulator, use a checkout with the
upstream commit available in Git history:

```bash
git show 2dd04f48f491adc545ba8e10663a458e153fbf85:flyhamlet/sim.py > /tmp/flyhamlet-sim-baseline.py
python scripts/check_prosthesis_regression.py --baseline /tmp/flyhamlet-sim-baseline.py
```

Implementation

A disjoint partition of selected read neurons supplies one count per channel per 10 ms bin. The lobe divides counts by population size and bin duration to obtain per-neuron mean rates, then clips rates/100 Hz to [0, 1]. Direct cue neurons remain excluded from all other ports in the benchmark. The core API permits purposeful read/write overlap, but rejects overlaps within a read port or within a write port.

For normalized input u, reservoir state h, and sampling interval D:

    alpha = 1 - exp(-D / tau)
    h_next = (1 - alpha) * h + alpha * tanh(W_in * u + W_rec * h)
    drive = clip(W_out * h_next + bias, -cap, cap)

The default reservoir has 256 units, tau = 500 ms, and a seeded orthogonal recurrent matrix scaled by 0.98. This keeps the isolated reservoir contractive; tanh and the leaky convex update bound its state. A reservoir reset erases state, partial spike bins, cached outputs, rate features, and diagnostic counters. Weights and explicit observation/feedback gates persist. The lobe uses its own initialization RNG and never draws from the fly's sensory RNG. No training update occurs during stepping.

A completed fly step reports its spikes to the lobe. At a completed sampling bin, the lobe computes its next output. That output can first enter the following timestep's synaptic slot. Each backend delivers prosthetic drive after native synapses and Poisson input and before spike reset:

    g[target] += (1 - exp(-dt / tau_syn)) * drive[target]

The simulator discards this increment on refractory targets. Its normal reset removes it on cells spiking in that step. This convention makes g approach the commanded drive in a nonspiking isolated cell and avoids an accidental 1/dt stimulation gain. The simulator's g has voltage-equivalent units. This interface does not model physical synaptic conductance, receptor dynamics, or a Dale-law-constrained new population. It leaves external current arrays and native synaptic weights unchanged.

The cap bounds each target's prosthetic contribution, not total network firing or the sum of native inputs. A circuit can amplify bounded stimulation. Full-brain experiments need population firing-rate monitoring, operating-range sweeps, and a predeclared stop rule for excessive activity. NumPy and Numba use CPU feedback; Torch transfers the sparse drive to its device. GPU correctness/performance was not tested. Default no-lobe execution follows the original code path except for disabled hooks.

Learning and evaluation

The initial write-back weights and bias are zero. Calibration runs independent balanced cue trials with feedback disabled. A ridge regression fits final delayed reservoir states to a two-channel stimulation target: cap for the cue's assigned write group and zero for the other group. This uses labels during supervised calibration only. The reservoir's input/recurrent weights and all anatomical synapses stay fixed. The calibration distribution can differ from the eventual closed-loop distribution; this prototype makes no optimality claim.

Each condition then receives separate trials to train a linear scoring decoder from late, downstream fly spike rates. A second scoring decoder reads the lobe state to measure what the external memory alone contains. Held-out trials change sensory-noise seeds, reset all dynamic state, and freeze every learned weight. Weight hashes check that evaluation preserves the trained lobe and native synapses. Evaluation labels serve only as the environment's sensory cue and subsequent score. Neither labels nor trial identifiers enter the lobe interface. Exact prediction ties select label 0 on a balanced schedule.

The trial sequence uses a 100 ms binary cue, 500 ms without that cue, and an additional 100 ms late scoring interval. Thus the scored activity occurs 500-600 ms after cue offset. Feedback, when enabled, operates throughout the trial. Ongoing stimulated activity can carry cue information during the blank interval; this experiment does not measure memory persistence after removing the prosthesis. Save/checkpoint operations retain weights and exact dense-index ports but intentionally exclude transient trial state. Full-run exports also record root IDs and a graph hash so ports can be audited against the graph.

Controls

Native: the fixed LIF network with an externally trained late-spike scoring decoder and no lobe.

Read-only: a recurrent lobe observes neural activity and retains state, with write-back disabled. This separates externally decodable memory from changes in downstream neural output.

Memoryless: the lobe replaces its state with the current bin's feedforward activation; both recurrent coupling and leaky history vanish. The native graph and its short-lived synaptic state still exist.

Leaky: the recurrent matrix is zero while the slow leaky state remains. This distinguishes the value of recurrent coupling from the value of a slow artificial time constant.

Closed-loop: the complete recurrent interface with learned, bounded write-back.

Disconnected: every native output synapse is silenced before training/evaluation. This tests dependence on graph-mediated routing. It does not test an advantage of biological anatomy over other graphs.

Observed demonstration

The synthetic graph has 64 excitatory LIF cells and 512 directed connected pairs. Eight cells per port and two sides provide disjoint cue, relay/read, write, and report populations. Engineered cue-to-relay and write-to-report paths support the task; weak write-to-relay paths connect the feedback interface. The graph contains no autonomous native recurrent loops. The toy's write cap is 12 mV, deliberately large enough to produce write-cell firing. The core's and full-cache runner's defaults use a 3 mV cap.

At seed 6, with 12 calibration trials per lobe condition and 12 decoder-training plus 24 evaluation trials for every condition, late-spike accuracy was 12/24 for native, read-only, memoryless, and disconnected runs, and 24/24 for leaky and closed-loop runs. The read-only, leaky, and closed-loop lobe-state decoders each scored 24/24. Memoryless and disconnected lobe-state decoders each scored 12/24. The two 24/24 feedback conditions show that slow state suffices in this engineered example; they supply no measured advantage for the recurrent matrix. Twenty-four trials from one synthetic graph do not quantify biological variability or broad task generalization.

Remaining empirical work

The 139,255-neuron FlyWire run, anatomical port selection, behavior/locomotion, GPU validation, and browser integration remain unperformed. The prototype supplies no observed improvement in the fly connectome. A stronger study should validate cue propagation before calibration, select ports using training data only, sweep unit counts including 1/2/8/32/256, sweep delay and stimulation amplitude, repeat across independent noise and initialization seeds, compare degree/sign-preserving rewired graphs, add a conventional synthetic controller, and lesion or scramble the read/write pathways. Such tests would separate externally added memory, neural routing, recurrent coupling, and any advantage of the anatomical wiring. Long-term plasticity inside FlyWire, reward-based learning, and learned locomotion need additional implementations.

Source and software provenance

The inspected upstream simulator has Git blob hash 9428cf8655b311c5c1e45a72c9803c5eabad88f1. The local copies of upstream data.py and config.py also matched their fetched Git blob hashes. Forty-six added tests passed on NumPy, Numba, and Torch CPU. Nine comparisons against the exact original simulator matched spikes, numerical state, RNG state, and reset/replay. These checks used synthetic graphs; the repository's complete existing test suite and GPU execution were not run. The regression script verifies the baseline Git blob hash; experiment result exports record source hashes and verification scope.
