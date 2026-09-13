# Experiment 1: typing entropy

4 flies, 1139 keystrokes in 1200 s of simulated time (0.949 keystrokes/s).

## Entropies (bits per keystroke)

| quantity | fly typist | uniform grid walk (null, 1e6 steps) | uniform 27-key typist (null) |
|---|---|---|---|
| H0 marginal, plug-in | 4.680 | 4.725 (exact) | 4.755 |
| H0 marginal, Miller-Madow | 4.696 | 4.725 | 4.755 |
| H(X_t+1 \| X_t), plug-in | 1.649 | 1.667 (exact 1.668) | 4.755 |
| H(X_t+1 \| X_t), Miller-Madow rows | 1.686 | | |
| LZ76 entropy rate (this length) | 1.890 | 2.451 | 4.163 |
| LZ76 entropy rate (2e5 symbols) | | 2.002 | |

The LZ76 estimate is upper-biased at short lengths, so the null models are also evaluated on sequences of the same length.

## Transition graph

- keys visited: 27 of 27; zero-visit keys: none
- strongly connected components among visited keys: 1 (one SCC: every visited key can reach every other)
- absorbing keys (visited, no exit observed): none

## Hamlet

Hamlet (Project Gutenberg #1524, lowercase a-z + space, whitespace collapsed): 167,325 characters, 167,324 transitions. 27^len = 10^239503.

Geometry: a keystroke is logged only on entering a new key region, so consecutive keys must be neighbours on the 3x9 grid (8-neighbourhood, corner crossings included). Of Hamlet's 468 distinct letter pairs only 105 are grid-adjacent under this layout (28,331 of 167,324 transitions). Every other pair has probability zero for any fly, however it moves: on a grid typewriter Hamlet is unreachable unless the layout makes all of its pairs adjacent.

Under the fitted first-order model P(Hamlet) = 0: 414 letter pairs needed by Hamlet were never typed (150,475 of Hamlet's transitions). Missing pairs and their frequencies in Hamlet (top 40):

| pair | count in Hamlet |
|---|---|
| `e ` | 5,652 |
| ` t` | 4,536 |
| `t ` | 4,238 |
| `th` | 3,985 |
| `s ` | 3,910 |
| `d ` | 3,038 |
| `he` | 2,962 |
| ` a` | 2,951 |
| ` h` | 2,611 |
| ` s` | 2,466 |
| ` i` | 2,277 |
| `n ` | 2,266 |
| `r ` | 2,167 |
| ` w` | 2,040 |
| `y ` | 2,032 |
| ` m` | 2,021 |
| `an` | 2,014 |
| `in` | 1,962 |
| `ou` | 1,962 |
| `o ` | 1,893 |
| `ha` | 1,871 |
| ` o` | 1,739 |
| `at` | 1,521 |
| `or` | 1,504 |
| `en` | 1,495 |
| ` b` | 1,405 |
| `is` | 1,335 |
| ` d` | 1,332 |
| ` l` | 1,232 |
| `le` | 1,205 |
| `on` | 1,199 |
| `it` | 1,179 |
| `l ` | 1,108 |
| `ar` | 1,088 |
| `es` | 1,074 |
| `h ` | 1,055 |
| ` c` | 1,041 |
| `ll` | 1,039 |
| `to` | 1,034 |
| ` n` | 967 |

With Laplace smoothing (pseudo-count 0.5): log2 P = -1041402, expected keystrokes ~ 10^313493.

## Figures

![transition heatmap](transition_heatmap.png)

![key visits](key_visit_heatmap.png)

![trajectory](trajectory.png)

