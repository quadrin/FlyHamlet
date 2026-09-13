# Two-letter learning benchmark: developmental smoke check

Generated 2026-09-13T21:20:57.847Z. 3 technical replicates of **one anatomical connectome** (FlyWire v783; 139,255 neurons, 15,091,983 connected pairs). This is a developmental smoke check, not a preregistered study or biological replication.

The unchanged default protocol supplies a left-eye cue for T or right-eye cue for O. Each trial resets neural state, presents 100 Hz Poisson stimulation for 200 ms, and reads seven downstream motor-group mean rates from the final 150 ms. An external logistic decoder receives 20 baseline, 60 supervised training, then 40 evaluation trials. Classes are balanced and independently shuffled within each phase. Evaluation contains fresh neural-noise seeds and no decoder updates. Cue information remains present: this tests classification/copying, not recall, autonomous typing, or biological synaptic learning.

| Seed | Baseline | Training, before each update | Evaluation | Evaluation 95% Wilson interval | Untrained control, evaluation | Compute time |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 42 | 55.0% | 63.3% | 100.0% (40/40) | 91.2%–100.0% | 65.0% | 86.85 s |
| 43 | 40.0% | 63.3% | 100.0% (40/40) | 91.2%–100.0% | 45.0% | 88.46 s |
| 44 | 55.0% | 58.3% | 100.0% (40/40) | 91.2%–100.0% | 60.0% | 88.72 s |

Intervals describe trial accuracy within each run. They are not population-level confidence intervals across brains. The untrained control uses unchanged zero weights with an independent, reproducible tie-breaking draw shared with the trained decoder on each trial. It receives the same neural features and targets for scoring; its weights never update. No shuffled-connectome, no-connectivity, or conventional-controller comparator is included, so these results do not establish an advantage of biological wiring.

All runs passed assertions that the evaluation weights equal their frozen start-of-evaluation values before and after **every** evaluation trial, no evaluation feedback is applied, exactly 60 supervised updates occur, and the control remains unchanged. Compressed and raw CSR files passed manifest SHA-256 checks. SHA-256 digests of every in-memory connectivity array remained identical after every run: no biological connection was edited.

Raw records (including per-trial neural seeds, feature vectors, decoder weights, correctness, provenance, protocol, and validation hashes):

- [Seed 42](seed-42.json)
- [Seed 43](seed-43.json)
- [Seed 44](seed-44.json)

Reproduce with Node.js:

```sh
node scripts/benchmark_learning.cjs --seeds 42,43,44 --out results/learning_benchmark
```
