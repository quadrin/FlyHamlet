# Two-letter state-decoding benchmark: developmental smoke check

Generated 2026-09-14T00:37:28.219Z. 3 seeded runs of one FlyWire v783 anatomical graph (139,255 neurons, 15,091,983 connection slots). Seeds are technical simulations, not independent biological animals.

Each condition receives the same balanced 64 training and 64 evaluation trials over the four ordered pairs tt, to, ot and oo. A 100 ms letter cue, a 25 ms gap and a second 100 ms letter cue end at 225 ms; all input is then switched OFF. One quantized snapshot of membrane voltage and synaptic current in 128 pools of nonstimulated neurons is read 25, 100, 200 ms later, at a measurement resolution of 0.01 mV per neuron. Two frozen ridge readouts per snapshot decode the first and second letters from the same 256 standardized features, with no external history, cue labels, clock, previous snapshot or correction. Exact-pair chance is 25%; remembering only the final letter gives 50% expected exact-pair accuracy and 50% first-letter accuracy.

The slower-dynamics condition multiplies the membrane and synaptic time constants by 10 (200 ms and 50 ms) and every signed synaptic weight by 0.1, so that each synaptic event keeps its integrated voltage response while its peak falls. Conduction delay, refractory period and thresholds are unchanged. This is an imposed engineering hypothesis, not measured fruit-fly physiology. The reset control uses the slower model and identical cues, then replaces all neural state at cue offset; its post-cue snapshots are exactly rest.

A separate diagnostic reads the same kind of snapshot at the final cue step, while input is still ON and before any reset. It tests encoding and is never combined with post-cue features. The delays are observations from the same trials and are correlated, not independent replications. Every condition/snapshot/position has its own scaler and ridge decoder fitted only on training observations, then frozen. No accuracy threshold or preferred outcome is enforced.

| Seed | Condition | Observation | Exact pair | First letter | Second letter | Mean neurons off rest (voltage) | Mean neurons off rest (current) | Mean spikes since OFF | Fully silent trials |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | original | Final cue step (input on) | 29/64 (45.3%) | 45.3% | 100.0% | 22417 | 17366 | 0.0 | 0/64 |
| 42 | original | 25 ms after OFF | 22/64 (34.4%) | 34.4% | 100.0% | 21902 | 8654 | 21.6 | 0/64 |
| 42 | original | 100 ms after OFF | 32/64 (50.0%) | 53.1% | 96.9% | 5830 | 0 | 22.5 | 0/64 |
| 42 | original | 200 ms after OFF | 23/64 (35.9%) | 40.6% | 82.8% | 3 | 0 | 22.5 | 10/64 |
| 42 | slow | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5962 | 7012 | 0.0 | 0/64 |
| 42 | slow | 25 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6371 | 7188 | 4.7 | 0/64 |
| 42 | slow | 100 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6488 | 5290 | 5.7 | 0/64 |
| 42 | slow | 200 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6109 | 1772 | 5.7 | 0/64 |
| 42 | reset | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5962 | 7012 | 0.0 | 0/64 |
| 42 | reset | 25 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 42 | reset | 100 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 42 | reset | 200 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 43 | original | Final cue step (input on) | 33/64 (51.6%) | 51.6% | 100.0% | 22437 | 17589 | 0.0 | 0/64 |
| 43 | original | 25 ms after OFF | 27/64 (42.2%) | 42.2% | 100.0% | 22087 | 8826 | 24.5 | 0/64 |
| 43 | original | 100 ms after OFF | 37/64 (57.8%) | 59.4% | 98.4% | 5969 | 0 | 25.6 | 0/64 |
| 43 | original | 200 ms after OFF | 22/64 (34.4%) | 42.2% | 85.9% | 5 | 0 | 25.6 | 8/64 |
| 43 | slow | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5907 | 6920 | 0.0 | 0/64 |
| 43 | slow | 25 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6323 | 7130 | 4.9 | 0/64 |
| 43 | slow | 100 ms after OFF | 63/64 (98.4%) | 98.4% | 100.0% | 6466 | 5267 | 5.9 | 0/64 |
| 43 | slow | 200 ms after OFF | 63/64 (98.4%) | 98.4% | 100.0% | 6105 | 1757 | 5.9 | 0/64 |
| 43 | reset | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5907 | 6920 | 0.0 | 0/64 |
| 43 | reset | 25 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 43 | reset | 100 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 43 | reset | 200 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 44 | original | Final cue step (input on) | 34/64 (53.1%) | 53.1% | 100.0% | 22480 | 16859 | 0.0 | 0/64 |
| 44 | original | 25 ms after OFF | 28/64 (43.8%) | 43.8% | 98.4% | 22038 | 9515 | 25.2 | 0/64 |
| 44 | original | 100 ms after OFF | 34/64 (53.1%) | 54.7% | 95.3% | 6117 | 0 | 26.6 | 0/64 |
| 44 | original | 200 ms after OFF | 23/64 (35.9%) | 48.4% | 79.7% | 5 | 0 | 26.6 | 10/64 |
| 44 | slow | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5934 | 6953 | 0.0 | 0/64 |
| 44 | slow | 25 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6326 | 7144 | 5.4 | 0/64 |
| 44 | slow | 100 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6467 | 5282 | 6.1 | 0/64 |
| 44 | slow | 200 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 6093 | 1731 | 6.1 | 0/64 |
| 44 | reset | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 5934 | 6953 | 0.0 | 0/64 |
| 44 | reset | 25 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 44 | reset | 100 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 44 | reset | 200 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |

Neurons "off rest" are nonstimulated neurons whose quantized voltage deviation or synaptic current is nonzero at the snapshot. Spikes since OFF count all neurons between input removal and the snapshot. Reading internal voltages and currents is more permissive than the spike-only Memory lab; a decodable subthreshold trace is not a demonstration of biologically accessible memory, learned motor control, language or memory for a text. Success under slower dynamics concerns that hypothetical model.

Validation independently rebuilt every scaler from training measurements, every feature from its snapshot and frozen scaler, and every evaluation choice from frozen weights, recomputed all pair, first-letter, second-letter and class-balanced scores, checked input-off timestamps, matched pair/noise schedules across conditions, confirmed identical slow/reset cue-period spikes and final-cue snapshots, and confirmed that reset post-cue snapshots are exactly rest. The 24 fitted readouts stayed frozen during evaluation. Original compressed/raw assets passed manifest SHA-256 checks; hashes of the anatomical graph and of the rescaled comparison graph (scaled weights SHA-256 d73770bbd383381b5de0ff8faf915972eb48f16f738c04de755c40dfff18f9b4) were unchanged after each complete run.

Compute times: seed 42: 357.62 s; seed 43: 358.05 s; seed 44: 358.87 s.

- [Seed 42: full records, models, provenance, and integrity checks (gzip JSON)](seed-42.json.gz)
- [Seed 43: full records, models, provenance, and integrity checks (gzip JSON)](seed-43.json.gz)
- [Seed 44: full records, models, provenance, and integrity checks (gzip JSON)](seed-44.json.gz)

```sh
node scripts/benchmark_sequence.cjs --seeds 42,43,44 --out results/sequence_benchmark
```
