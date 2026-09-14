# Self-driven phrase recall benchmark: developmental smoke check

Generated 2026-09-14T01:12:02.145Z. 1 seeded run of one FlyWire v783 anatomical graph (139,255 neurons, 15,091,983 connection slots). Seeds are technical simulations, not independent biological animals.

Reference phrase: `to be or not to be` (18 characters plus END). Each condition trains one scaler and one ridge readout on 6 teacher-forced episodes (114 snapshots), then runs 3 held-out teacher-forced diagnostic episodes and 6 autonomous recall episodes. Every cue lasts 100 ms at 100 Hz, followed by a 25 ms gap; the snapshot of 256 quantized voltage and current pool means (0.01 mV per neuron) is read at the end of the gap and the next cue starts immediately. In retained conditions one network runs through the whole episode without reset. During recall the decoded letter becomes the next sensory cue; END stops an episode and a cap of 32 decisions does not depend on the phrase length. Chain length is the longest correct prefix of the output.

The slower-dynamics condition multiplies the membrane and synaptic time constants by 10 (200 ms and 50 ms) and every signed synaptic weight by 0.1; delay, refractory period and thresholds are unchanged. It is an imposed engineering hypothesis. The reset control uses the slower model and identical cues but replaces all neural state at every cue onset, so each snapshot reflects only the current letter. A current-cue-only comparator (ridge on the one-hot cue, own-feedback rollout, no state) produces `to␣be␣be␣be␣be␣be␣be␣be␣be␣be␣be` with chain length 6: the best any decoder can do without memory of earlier letters.

| Seed | Condition | Diagnostic next-letter accuracy (teacher-forced, held out) | Recall chain lengths | Mean chain | Exact | Mean edit distance | Mean spikes per recall cue | Mean neurons off rest at recall snapshots |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | original | 14/57 (24.6%) | 2, 1, 0, 1, 0, 1 | 0.8 | 0/6 | 16.2 | 692 | 22207 |
| 42 | slow | 55/57 (96.5%) | 5, 5, 5, 18, 10, 18 | 10.2 | 1/6 | 10.0 | 407 | 10105 |
| 42 | reset | 23/57 (40.4%) | 2, 1, 2, 2, 3, 1 | 1.8 | 0/6 | 12.3 | 394 | 4152 |

| Seed | Condition | Episode | Autonomous output |
| --- | --- | ---: | --- |
| 42 | original | 1 | `tototnoobe␣nor␣o␣t␣t␣be␣nono␣to` |
| 42 | original | 2 | `t␣no␣ototototbtoo` |
| 42 | original | 3 | `` |
| 42 | original | 4 | `t␣t␣oto␣bntootr␣b␣eto␣obe␣␣nteet` |
| 42 | original | 5 | `␣o␣otnto␣` |
| 42 | original | 6 | `ttoboo␣to␣tbe␣r␣e␣etobe␣to␣rno␣r` |
| 42 | slow | 1 | `to␣be` |
| 42 | slow | 2 | `to␣be` |
| 42 | slow | 3 | `to␣be` |
| 42 | slow | 4 | `to␣be␣or␣not␣to␣be␣or␣not␣to␣be` |
| 42 | slow | 5 | `to␣be␣or␣n` |
| 42 | slow | 6 | `to␣be␣or␣not␣to␣be` |
| 42 | reset | 1 | `totototo␣␣␣␣r␣be␣t␣ber␣rt␣ber␣tt` |
| 42 | reset | 2 | `t␣tor␣o␣otor␣be` |
| 42 | reset | 3 | `tor␣be␣␣o␣be␣be` |
| 42 | reset | 4 | `tor␣or␣be␣ttor␣be` |
| 42 | reset | 5 | `to␣␣notor␣tottoto␣o␣be␣␣␣o␣be␣ot` |
| 42 | reset | 6 | `t␣otoo␣or␣be` |

Validation independently rebuilt every scaler from training measurements, every feature from its snapshot and frozen scaler, and every choice from frozen weights; confirmed that every recall cue was the actor's own previous prediction and never the reference; recomputed chain lengths, edit distances and diagnostic accuracies; and confirmed that teacher-forced phases share cues and episode seeds across conditions. The three fitted readouts stayed frozen. Original compressed/raw assets passed manifest SHA-256 checks; hashes of the anatomical graph and of the rescaled comparison graph (scaled weights SHA-256 d73770bbd383381b5de0ff8faf915972eb48f16f738c04de755c40dfff18f9b4) were unchanged after each complete run. Reading internal voltages is more permissive than spike decoding; a long chain under slower dynamics concerns that hypothetical model, not fruit-fly physiology, and no result here is memory for a text.

Compute times: seed 42: 352.83 s.

- [Seed 42: full records, models, provenance, and integrity checks (gzip JSON)](seed-42.json.gz)

```sh
node scripts/benchmark_chain.cjs --seeds 42 --out results/chain_benchmark
```
