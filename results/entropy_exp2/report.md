# Experiment 2: entropy estimates (NIST SP 800-90B)

3,603,323 raw ISIs (timesteps) from 4 file(s).

| data | estimator | min-entropy (bits/sample) | notes |
|---|---|---|---|
| raw ISI symbols | MCV | 7.545 | alphabet 26,477, mode 45 (p=0.0053) |
| raw ISI bits (21-bit) | MCV | 0.341 | per bit |
| raw ISI bits (21-bit) | collision | 0.203 | per bit, p=0.8688 |
| raw ISI low byte | MCV | 7.149 | per byte (max 8) |
| raw ISI low byte bits | collision | 1.000 | per bit |
| whitened (SHA-256) bytes | MCV | 7.910 | per byte (max 8) |
| whitened (SHA-256) bits | collision | 0.944 | per bit |

dieharder input written to `results/entropy_exp2/raw.*` (uint32 ISIs) and `results/entropy_exp2/whitened.*` (SHA-256 stream): `dieharder -a -g 201 -f <file>`.

## Tap and whitening

4 flies x 120 s in the arena with Poisson background drive (200 Hz x 4 mV on every neuron) plus
the looming inputs; 500 tapped central-brain neurons per fly (lists in `results/entropy/isi_flyNN_neurons.csv`),
~430 of which fire (median 2.6 Hz, mean 15 Hz); 3,603,323 ISIs in total. Whitening: 56,301 batches of
64 ISIs -> SHA-256 -> 1,801,632 bytes -> 1,709,435 keys after rejecting bytes >= 243 (accept fraction 0.949 = 243/256).

## Typist (suffix automaton over Hamlet)

```
[key         1] new record:   1 chars 's' (Hamlet offset 51)
[key         2] new record:   2 chars 's ' (Hamlet offset 70)
[key         3] new record:   3 chars 's a' (Hamlet offset 70)
[key         4] new record:   4 chars 's ab' (Hamlet offset 7,781)
[key       454] new record:   5 chars 'ng th' (Hamlet offset 2,620)
[key     3,619] new record:   6 chars 'ng let' (Hamlet offset 71,752)
[key    26,235] new record:   7 chars 'c look ' (Hamlet offset 93,340)
[key    26,236] new record:   8 chars 'c look y' (Hamlet offset 93,340)
keys consumed: 1,709,435; longest Hamlet substring: 8 chars 'c look y' (at key 26,236, Hamlet offset 93,340); 2,265,270 keys/s
```

The record grows roughly logarithmically with the number of keys, as expected for an i.i.d.
uniform source: 27^8 ~ 2.8e11, and a 167k-character text offers ~167k starting points, so an
8-character match is expected after ~1e6 keys and a 9-character one after ~3e7.

## dieharder

Subset sts_monobit, sts_runs, sts_serial, diehard_runs, operm5, rank32x32 (`scripts/run_dieharder.sh`, `dieharder.txt`):

| stream | FAILED | WEAK | PASSED |
|---|---|---|---|
| raw ISIs (uint32) | 34 | 2 | 0 |
| whitened SHA-256 | 11 | 11 | 14 |
| /dev/urandom control, same size (1.8 MB) | 9 | 13 | 12 |

The whitened stream and the true-random control have the same profile: dieharder needs far more
than 1.8 MB and rewinds the file many times per test, which by itself produces FAILED/WEAK
verdicts. All 56,301 SHA-256 digests are distinct (byte mean 127.54). The raw ISIs fail on their
own merits (high bits mostly zero, strongly non-uniform), which is why they are hashed.
