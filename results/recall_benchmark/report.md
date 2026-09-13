# Phrase-recall benchmark: developmental smoke check

Generated 2026-09-13T21:39:35.253Z. 3 seeded runs of **one anatomical connectome** (FlyWire v783; 139,255 neurons, 15,091,983 connected pairs). Repeated seeds and rollouts are technical simulations, not independent biological brains. This is a developmental smoke check, not a preregistered study.

Reference phrase: **to be or not to be**. Each run uses the unchanged default protocol: 6 teacher-forced training episodes, then 3 autonomous rollouts and 3 rollouts with past history masked. During autonomous recall, the next cue comes exclusively from the preceding emitted character, beginning with START; neither a next-letter cue nor a correction is supplied. END is a learned output. Exact success requires both the literal reference phrase and a predicted END. Incorrect outputs and decisions remain in the raw records.

The full connectome is fixed. Learning occurs in an external decoder, with an external history buffer of 6 feature vectors. The ablation keeps the current 128-component neural vector and masks all 5 past vectors; it uses the same trained weights and self-generated cues. Symbol cues are arbitrary experimenter-designed codes, not a claim that the fly recognizes written characters. The conventional comparator learns from a separate history of explicit character one-hot vectors on the same training examples and then generates its own feedback. Any successful recall is a property of the combined model, external memory, coding scheme, and decoder; it does not demonstrate biological synaptic learning, Shakespeare comprehension, or recall from fly wiring alone.

| Seed | Condition | Exact phrase + END | Ended with END | Capped episodes | Mean edit distance |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | recall | 3/3 | 3/3 | 0 | 0.00 |
| 42 | ablated | 0/3 | 0/3 | 3 | 45.00 |
| 42 | comparator | 3/3 | 3/3 | 0 | 0.00 |
| 43 | recall | 3/3 | 3/3 | 0 | 0.00 |
| 43 | ablated | 0/3 | 0/3 | 3 | 45.00 |
| 43 | comparator | 3/3 | 3/3 | 0 | 0.00 |
| 44 | recall | 3/3 | 3/3 | 0 | 0.00 |
| 44 | ablated | 0/3 | 0/3 | 3 | 45.00 |
| 44 | comparator | 3/3 | 3/3 | 0 | 0.00 |

Compute time: Seed 42: 103.90 s; Seed 43: 110.35 s; Seed 44: 102.18 s. No minimum accuracy or preferred outcome is enforced. Edit distances and stopping success above are independently recomputed from exported decisions. Repeated conventional episodes are identical deterministic rollouts, not independent evidence. The ablation and conventional comparator are limited controls, not a matched shuffled-connectome experiment or proof of an advantage from anatomical wiring.

All runs passed assertions that autonomous inputs match preceding self-generated choices, no autonomous record contains a teacher target, outputs reconstruct exactly from the chosen characters, END terminates generation, and final decoder weights equal their frozen values. Compressed and raw CSR files passed the manifest SHA-256 checks. In-memory hashes of every connectivity array were unchanged after each run.

Raw records include neural seeds, trial features, generated decisions, stopping reasons, model/protocol provenance, fitted weights, and validation hashes:

- [Seed 42](seed-42.json)
- [Seed 43](seed-43.json)
- [Seed 44](seed-44.json)

Reproduce with Node.js:

```sh
node scripts/benchmark_recall.cjs --seeds 42,43,44 --out results/recall_benchmark
```
