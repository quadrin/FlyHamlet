# Post-cue neural-memory benchmark: developmental smoke check

Generated 2026-09-13T22:11:43.371Z. 3 seeded runs of one FlyWire v783 anatomical graph (139,255 neurons, 15,091,983 connection slots) and one fixed rewired null (seed 1299709). Seeds are technical simulations, not independent biological animals or independently drawn null graphs.

Each condition receives the same balanced 56 training and 56 evaluation trials over seven arbitrary sensory symbols. A 200 ms cue is switched OFF; six independent decoders read 20 ms spike windows beginning 0, 10, 25, 50, 100, 200 ms later. Each decoder receives only its own window's 128 pooled neural rates, with no external history, cue labels, previous observations, or corrections. State, delayed events, and noise state are reset at cue offset in the reset condition. The retained and rewired conditions keep their neural state. Chance classification is 1/7 (14.3%).

A seventh, separate decoder reads the final 20 ms while the cue is still ON. This diagnostic tests initial cue encoding and is never merged into a post-cue feature vector or memory score. Differences in cue-on accuracy indicate different initial encoding, complicating attribution of later differences specifically to retention.

The six delays are observations from the same trials; some windows overlap. They are correlated measurements, not six independent replications. Every condition/delay has its own ridge decoder fitted only on training observations, then frozen. No accuracy threshold or preferred biological outcome is enforced.

| Seed | Condition | Observation | Accuracy | Balanced accuracy | Mean population spikes / window | Fully silent trials | Readout silent trials |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | retain | Cue ON diagnostic | 25/56 (44.6%) | 44.6% | 136.1 | 0/56 | 0/56 |
| 42 | retain | 0 ms after OFF | 13/56 (23.2%) | 23.2% | 23.3 | 0/56 | 0/56 |
| 42 | retain | 10 ms after OFF | 9/56 (16.1%) | 16.1% | 5.6 | 19/56 | 19/56 |
| 42 | retain | 25 ms after OFF | 9/56 (16.1%) | 16.1% | 1.4 | 40/56 | 40/56 |
| 42 | retain | 50 ms after OFF | 7/56 (12.5%) | 12.5% | 0.1 | 55/56 | 55/56 |
| 42 | retain | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | retain | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | Cue ON diagnostic | 25/56 (44.6%) | 44.6% | 136.1 | 0/56 | 0/56 |
| 42 | reset | 0 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | reset | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | rewired | Cue ON diagnostic | 49/56 (87.5%) | 87.5% | 85.0 | 0/56 | 0/56 |
| 42 | rewired | 0 ms after OFF | 20/56 (35.7%) | 35.7% | 2.3 | 6/56 | 12/56 |
| 42 | rewired | 10 ms after OFF | 9/56 (16.1%) | 16.1% | 0.0 | 55/56 | 55/56 |
| 42 | rewired | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | rewired | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | rewired | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 42 | rewired | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | retain | Cue ON diagnostic | 33/56 (58.9%) | 58.9% | 138.9 | 0/56 | 0/56 |
| 43 | retain | 0 ms after OFF | 15/56 (26.8%) | 26.8% | 25.8 | 0/56 | 0/56 |
| 43 | retain | 10 ms after OFF | 10/56 (17.9%) | 17.9% | 5.8 | 14/56 | 14/56 |
| 43 | retain | 25 ms after OFF | 9/56 (16.1%) | 16.1% | 1.0 | 46/56 | 46/56 |
| 43 | retain | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | retain | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | retain | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | Cue ON diagnostic | 33/56 (58.9%) | 58.9% | 138.9 | 0/56 | 0/56 |
| 43 | reset | 0 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | reset | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | rewired | Cue ON diagnostic | 54/56 (96.4%) | 96.4% | 85.4 | 0/56 | 0/56 |
| 43 | rewired | 0 ms after OFF | 25/56 (44.6%) | 44.6% | 2.5 | 5/56 | 14/56 |
| 43 | rewired | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 55/56 | 55/56 |
| 43 | rewired | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | rewired | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | rewired | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 43 | rewired | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | retain | Cue ON diagnostic | 33/56 (58.9%) | 58.9% | 137.4 | 0/56 | 0/56 |
| 44 | retain | 0 ms after OFF | 12/56 (21.4%) | 21.4% | 25.3 | 0/56 | 0/56 |
| 44 | retain | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 6.6 | 9/56 | 9/56 |
| 44 | retain | 25 ms after OFF | 5/56 (8.9%) | 8.9% | 2.0 | 39/56 | 39/56 |
| 44 | retain | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.1 | 54/56 | 54/56 |
| 44 | retain | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | retain | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | Cue ON diagnostic | 33/56 (58.9%) | 58.9% | 137.4 | 0/56 | 0/56 |
| 44 | reset | 0 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | reset | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | rewired | Cue ON diagnostic | 49/56 (87.5%) | 87.5% | 84.5 | 0/56 | 0/56 |
| 44 | rewired | 0 ms after OFF | 25/56 (44.6%) | 44.6% | 2.1 | 9/56 | 12/56 |
| 44 | rewired | 10 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | rewired | 25 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | rewired | 50 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | rewired | 100 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |
| 44 | rewired | 200 ms after OFF | 8/56 (14.3%) | 14.3% | 0.0 | 56/56 | 56/56 |

Population activity includes all simulated cells. Readout silence refers only to nonstimulated neurons used by the pooled decoder. All memory observations occur after input removal; delayed synaptic events already in flight remain part of retained neural state. Silence or failed decoding in this LIF experiment does not show that biological flies cannot remember. Successful decoding is not a demonstration of phrase memory.

The null randomly permutes target slots while retaining every source slot and its signed weight. Incoming and outgoing connection-slot degrees are preserved for every neuron. It allows self-loops and parallel slots: 288 self-loop slots and 57,897 parallel slots, leaving 15,034,086 distinct directed pairs. Incoming weighted strength, per-neuron excitation/inhibition balance, spatial geometry, and numbers of distinct neighbors are not matched. One null realization cannot establish a population-wide effect of wiring.

Validation independently reconstructed every feature from its window's spike counts and every evaluation score from frozen weights, recomputed all accuracies and class-balanced scores, checked cue-OFF timestamps, matched input/noise schedules across controls, and confirmed identical retain/reset cue-period counts. The 21 fitted readouts stayed frozen during evaluation. Original compressed/raw assets passed manifest SHA-256 checks; hashes of both in-memory graphs were unchanged after each complete run.

Compute times: seed 42: 204.50 s; seed 43: 244.53 s; seed 44: 245.50 s.

- [Seed 42: full records, models, provenance, and integrity checks (gzip JSON)](seed-42.json.gz)
- [Seed 43: full records, models, provenance, and integrity checks (gzip JSON)](seed-43.json.gz)
- [Seed 44: full records, models, provenance, and integrity checks (gzip JSON)](seed-44.json.gz)

```sh
node scripts/benchmark_memory.cjs --seeds 42,43,44 --out results/memory_benchmark
```
