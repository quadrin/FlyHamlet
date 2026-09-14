# In-brain mushroom-body memory benchmark: developmental smoke check

Generated 2026-09-14T02:21:48.365Z. 3 seeded runs of one FlyWire v783 anatomical graph (139,255 neurons, 15,091,983 connection slots) with 5,177 Kenyon cells, 96 MBONs and 277 uniglomerular projection neurons from the public classification table (annotation SHA-256 c79759d2be5b4f053a95d1bb7f1d9e8977aaa7119836b66fe5f3e406693350ab). Seeds are technical simulations, not independent biological animals.

Reference phrase: `to be or not to be`. Letters are sparse codes on disjoint seeded sets of projection neurons (100 Hz for 100 ms, then a 25 ms gap; the first cue of an episode, and every cue in the reset condition, is preceded by a 250 ms warm-up of the same cue outside the counting window). The readout is a fixed seeded partition of the MBONs into eight groups; the decision is the group with the most spikes in the 125 ms window. The only learning is a supervised three-factor rule on the 62,261 existing Kenyon-cell-to-MBON synapses during 12 teacher-forced training episodes (1 mV per Kenyon-cell spike, bounded to [0, 20] mV), applied only when the target group is not the strict winner. Learning is off for 3 teacher-forced diagnostic episodes and 6 autonomous recall episodes, in which the decoded letter becomes the next cue; END stops an episode and a cap of 32 decisions does not depend on the phrase length. Chain length is the longest correct prefix.

The model multiplies time constants by 10 (200 ms and 50 ms) and all weights by 0.3, a calibration at which a letter activates a few percent of Kenyon cells with letter- and position-dependent codes; the original weights saturate two thirds of all Kenyon cells for every letter. The plastic condition keeps one network per episode; the reset condition replaces all neural state at every cue onset; the frozen condition never learns. A current-cue-only comparator produces `to␣be␣be␣be␣be␣be␣be␣be␣be␣be␣be` with chain length 6: the best any decoder can do without memory of earlier letters.

| Seed | Condition | Training accuracy by episode | Updates | Diagnostic accuracy (learning off) | Recall chain lengths | Mean chain | Exact | Synapses changed | Mean weight (mV) | MBON spikes per recall cue | Kenyon cells per recall cue |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | plastic | 26.3%, 47.4%, 52.6%, 42.1%, 42.1%, 47.4%, 63.2%, 47.4%, 68.4%, 57.9%, 73.7%, 68.4% | 121 | 45/57 (78.9%) | 5, 5, 2, 5, 5, 5 | 4.5 | 0/6 | 22,842 | 0.68 | 56.8 | 8.5% |
| 42 | plasticReset | 0.0%, 42.1%, 42.1%, 36.8%, 42.1%, 31.6%, 47.4%, 42.1%, 52.6%, 52.6%, 47.4%, 42.1% | 142 | 34/57 (59.6%) | 6, 5, 5, 5, 6, 5 | 5.3 | 0/6 | 23,229 | 0.72 | 64.3 | 11.0% |
| 42 | frozen | 15.8%, 10.5%, 5.3%, 5.3%, 5.3%, 5.3%, 15.8%, 5.3%, 15.8%, 15.8%, 10.5%, 10.5% | 0 | 5/57 (8.8%) | 0, 0, 0, 0, 0, 0 | 0.0 | 0/6 | 0 | 0.34 | 10.9 | 10.5% |
| 43 | plastic | 15.8%, 52.6%, 47.4%, 42.1%, 52.6%, 47.4%, 47.4%, 63.2%, 52.6%, 63.2%, 47.4%, 57.9% | 131 | 44/57 (77.2%) | 0, 6, 0, 7, 5, 5 | 3.8 | 0/6 | 22,648 | 0.70 | 60.7 | 8.3% |
| 43 | plasticReset | 5.3%, 36.8%, 47.4%, 42.1%, 47.4%, 47.4%, 52.6%, 47.4%, 47.4%, 68.4%, 36.8%, 36.8% | 136 | 35/57 (61.4%) | 5, 5, 5, 5, 5, 6 | 5.2 | 0/6 | 23,452 | 0.71 | 62.9 | 11.0% |
| 43 | frozen | 10.5%, 5.3%, 10.5%, 10.5%, 5.3%, 15.8%, 15.8%, 10.5%, 21.1%, 5.3%, 5.3%, 10.5% | 0 | 4/57 (7.0%) | 0, 0, 0, 0, 0, 0 | 0.0 | 0/6 | 0 | 0.34 | 11.1 | 10.6% |
| 44 | plastic | 21.1%, 42.1%, 36.8%, 57.9%, 63.2%, 68.4%, 57.9%, 52.6%, 47.4%, 57.9%, 52.6%, 73.7% | 120 | 38/57 (66.7%) | 5, 5, 0, 0, 0, 6 | 2.7 | 0/6 | 22,407 | 0.68 | 57.1 | 8.5% |
| 44 | plasticReset | 0.0%, 36.8%, 42.1%, 42.1%, 36.8%, 42.1%, 36.8%, 42.1%, 52.6%, 36.8%, 31.6%, 42.1% | 150 | 36/57 (63.2%) | 5, 0, 5, 6, 5, 5 | 4.3 | 0/6 | 24,752 | 0.75 | 66.3 | 10.8% |
| 44 | frozen | 5.3%, 10.5%, 10.5%, 5.3%, 15.8%, 5.3%, 10.5%, 0.0%, 10.5%, 10.5%, 0.0%, 5.3% | 0 | 3/57 (5.3%) | 0, 0, 0, 0, 0, 0 | 0.0 | 0/6 | 0 | 0.34 | 10.9 | 10.4% |

| Seed | Condition | Episode | Autonomous output |
| --- | --- | ---: | --- |
| 42 | plastic | 1 | `to␣be` |
| 42 | plastic | 2 | `to␣be` |
| 42 | plastic | 3 | `tot␣be` |
| 42 | plastic | 4 | `to␣be` |
| 42 | plastic | 5 | `to␣be` |
| 42 | plastic | 6 | `to␣be` |
| 42 | plasticReset | 1 | `to␣be␣be` |
| 42 | plasticReset | 2 | `to␣be` |
| 42 | plasticReset | 3 | `to␣be` |
| 42 | plasticReset | 4 | `to␣be` |
| 42 | plasticReset | 5 | `to␣be␣be` |
| 42 | plasticReset | 6 | `to␣be` |
| 42 | frozen | 1 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 42 | frozen | 2 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 42 | frozen | 3 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 42 | frozen | 4 | `bbbbbbbbbbbbnbbbbbbbbbbbbnbbbbbb` |
| 42 | frozen | 5 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 42 | frozen | 6 | `bbbbbbbbbbbbbbbbbbbbbbnbbbbbbbbb` |
| 43 | plastic | 1 | `␣be` |
| 43 | plastic | 2 | `to␣be␣be` |
| 43 | plastic | 3 | `␣no␣be␣oto␣be␣or␣be` |
| 43 | plastic | 4 | `to␣be␣o␣be` |
| 43 | plastic | 5 | `to␣be` |
| 43 | plastic | 6 | `to␣be` |
| 43 | plasticReset | 1 | `to␣be` |
| 43 | plasticReset | 2 | `to␣be` |
| 43 | plasticReset | 3 | `to␣be` |
| 43 | plasticReset | 4 | `to␣be` |
| 43 | plasticReset | 5 | `to␣be` |
| 43 | plasticReset | 6 | `to␣be␣be` |
| 43 | frozen | 1 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 43 | frozen | 2 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 43 | frozen | 3 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 43 | frozen | 4 | `bbbbbbbbbbbbbbbbbbbbbbbbbbtbbbbb` |
| 43 | frozen | 5 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 43 | frozen | 6 | `bbbbbbbbbbbbbbbbbbbbbbbbbnbbbbbb` |
| 44 | plastic | 1 | `to␣be` |
| 44 | plastic | 2 | `to␣be` |
| 44 | plastic | 3 | `be` |
| 44 | plastic | 4 | `␣be` |
| 44 | plastic | 5 | `␣be␣be` |
| 44 | plastic | 6 | `to␣be␣be␣be` |
| 44 | plasticReset | 1 | `to␣be` |
| 44 | plasticReset | 2 | `␣be` |
| 44 | plasticReset | 3 | `to␣be` |
| 44 | plasticReset | 4 | `to␣be␣be` |
| 44 | plasticReset | 5 | `to␣be` |
| 44 | plasticReset | 6 | `to␣be` |
| 44 | frozen | 1 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 44 | frozen | 2 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbnbbb` |
| 44 | frozen | 3 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 44 | frozen | 4 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 44 | frozen | 5 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |
| 44 | frozen | 6 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |

Validation rebuilt the plastic slot list from the graph and exported groups, confirmed that initial plastic weights are the rescaled anatomical weights, replayed every synaptic update from the recorded Kenyon-cell activity and decisions to reproduce the learned weights exactly, confirmed that learning occurred only during training of the plastic conditions and that the frozen control's synapses never changed, reproduced every decision from the exported group counts and tie scores, confirmed that every recall cue was the actor's own previous prediction, and recomputed chain lengths, edit distances and accuracies. Hashes of the anatomical graph and of the rescaled graph (scaled weights SHA-256 3c085000c0fb715e9ae0c596169a0c51311d1319d10ff442ebe41ea9f2090227) were unchanged after each complete run; experiments write only to their private copy of the weight array. The model, the codes and the teacher are imposed; nothing here shows how flies learn, and a recited phrase is not memory for a text.

Compute times: seed 42: 1292.45 s; seed 43: 1306.64 s; seed 44: 1289.07 s.

- [Seed 42: full records, weights, provenance, and integrity checks (gzip JSON)](seed-42.json.gz)
- [Seed 43: full records, weights, provenance, and integrity checks (gzip JSON)](seed-43.json.gz)
- [Seed 44: full records, weights, provenance, and integrity checks (gzip JSON)](seed-44.json.gz)

```sh
node scripts/benchmark_mb.cjs --seeds 42,43,44 --out results/mb_benchmark
```
