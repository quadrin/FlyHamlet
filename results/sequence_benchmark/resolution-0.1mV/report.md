# Two-letter state-decoding benchmark: developmental smoke check

Generated 2026-09-14T00:47:47.978Z. 1 seeded run of one FlyWire v783 anatomical graph (139,255 neurons, 15,091,983 connection slots). Seeds are technical simulations, not independent biological animals.

Each condition receives the same balanced 64 training and 64 evaluation trials over the four ordered pairs tt, to, ot and oo. A 100 ms letter cue, a 25 ms gap and a second 100 ms letter cue end at 225 ms; all input is then switched OFF. One quantized snapshot of membrane voltage and synaptic current in 128 pools of nonstimulated neurons is read 25, 100, 200 ms later, at a measurement resolution of 0.1 mV per neuron. Two frozen ridge readouts per snapshot decode the first and second letters from the same 256 standardized features, with no external history, cue labels, clock, previous snapshot or correction. Exact-pair chance is 25%; remembering only the final letter gives 50% expected exact-pair accuracy and 50% first-letter accuracy.

The slower-dynamics condition multiplies the membrane and synaptic time constants by 10 (200 ms and 50 ms) and every signed synaptic weight by 0.1, so that each synaptic event keeps its integrated voltage response while its peak falls. Conduction delay, refractory period and thresholds are unchanged. This is an imposed engineering hypothesis, not measured fruit-fly physiology. The reset control uses the slower model and identical cues, then replaces all neural state at cue offset; its post-cue snapshots are exactly rest.

A separate diagnostic reads the same kind of snapshot at the final cue step, while input is still ON and before any reset. It tests encoding and is never combined with post-cue features. The delays are observations from the same trials and are correlated, not independent replications. Every condition/snapshot/position has its own scaler and ridge decoder fitted only on training observations, then frozen. No accuracy threshold or preferred outcome is enforced.

| Seed | Condition | Observation | Exact pair | First letter | Second letter | Mean neurons off rest (voltage) | Mean neurons off rest (current) | Mean spikes since OFF | Fully silent trials |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | original | Final cue step (input on) | 31/64 (48.4%) | 48.4% | 100.0% | 16486 | 12384 | 0.0 | 0/64 |
| 42 | original | 25 ms after OFF | 22/64 (34.4%) | 35.9% | 98.4% | 12184 | 3110 | 21.6 | 0/64 |
| 42 | original | 100 ms after OFF | 30/64 (46.9%) | 53.1% | 90.6% | 1080 | 0 | 22.5 | 0/64 |
| 42 | original | 200 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 22.5 | 64/64 |
| 42 | slow | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 2791 | 4159 | 0.0 | 0/64 |
| 42 | slow | 25 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 3038 | 3995 | 4.7 | 0/64 |
| 42 | slow | 100 ms after OFF | 64/64 (100.0%) | 100.0% | 100.0% | 2936 | 1445 | 5.7 | 0/64 |
| 42 | slow | 200 ms after OFF | 63/64 (98.4%) | 98.4% | 100.0% | 1989 | 288 | 5.7 | 0/64 |
| 42 | reset | Final cue step (input on) | 64/64 (100.0%) | 100.0% | 100.0% | 2791 | 4159 | 0.0 | 0/64 |
| 42 | reset | 25 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 42 | reset | 100 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |
| 42 | reset | 200 ms after OFF | 16/64 (25.0%) | 50.0% | 50.0% | 0 | 0 | 0.0 | 64/64 |

Neurons "off rest" are nonstimulated neurons whose quantized voltage deviation or synaptic current is nonzero at the snapshot. Spikes since OFF count all neurons between input removal and the snapshot. Reading internal voltages and currents is more permissive than the spike-only Memory lab; a decodable subthreshold trace is not a demonstration of biologically accessible memory, learned motor control, language or memory for a text. Success under slower dynamics concerns that hypothetical model.

Validation independently rebuilt every scaler from training measurements, every feature from its snapshot and frozen scaler, and every evaluation choice from frozen weights, recomputed all pair, first-letter, second-letter and class-balanced scores, checked input-off timestamps, matched pair/noise schedules across conditions, confirmed identical slow/reset cue-period spikes and final-cue snapshots, and confirmed that reset post-cue snapshots are exactly rest. The 24 fitted readouts stayed frozen during evaluation. Original compressed/raw assets passed manifest SHA-256 checks; hashes of the anatomical graph and of the rescaled comparison graph (scaled weights SHA-256 d73770bbd383381b5de0ff8faf915972eb48f16f738c04de755c40dfff18f9b4) were unchanged after each complete run.

Compute times: seed 42: 355.56 s.

- [Seed 42: full records, models, provenance, and integrity checks (gzip JSON)](seed-42.json.gz)

```sh
node scripts/benchmark_sequence.cjs --seeds 42 --out results/sequence_benchmark --resolution 0.1
```
