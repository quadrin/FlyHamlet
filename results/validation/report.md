# Phase 1 validation: sugar GRN -> feeding motor neurons

20 labellar sugar GRNs (v783 side = left) driven by Poisson input; 5 trials x 1.0 s per rate; mean +/- s.d. across trials. Shiu et al. 2024 parameters (w_syn = 0.275 mV).

## Motor neuron firing rates (Hz)

| GRN rate | MN9 ipsi | MN9 contra | MN6 ipsi | MN6 contra | MN8 ipsi | MN8 contra | responders (>0 Hz) |
|---|---|---|---|---|---|---|---|
| 10 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 28 |
| 25 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 63 |
| 50 | 7.6 +/- 2.8 | 10.2 +/- 3.1 | 2.2 +/- 3.9 | 3.4 +/- 3.4 | 14.4 +/- 2.7 | 0.6 +/- 1.2 | 289 |
| 75 | 41.2 +/- 2.7 | 50.0 +/- 4.6 | 19.4 +/- 3.0 | 24.2 +/- 2.1 | 48.2 +/- 3.4 | 14.8 +/- 0.7 | 358 |
| 100 | 47.4 +/- 3.3 | 62.8 +/- 6.1 | 25.6 +/- 2.2 | 28.0 +/- 2.1 | 58.2 +/- 4.0 | 22.4 +/- 2.7 | 374 |
| 125 | 51.6 +/- 2.2 | 70.4 +/- 2.7 | 26.6 +/- 2.2 | 33.0 +/- 1.9 | 68.8 +/- 2.6 | 26.0 +/- 1.7 | 391 |
| 150 | 56.8 +/- 2.9 | 79.2 +/- 5.0 | 29.0 +/- 2.0 | 35.8 +/- 1.2 | 75.6 +/- 2.6 | 30.6 +/- 2.8 | 393 |
| 175 | 61.2 +/- 3.9 | 83.6 +/- 4.5 | 32.6 +/- 3.4 | 39.0 +/- 1.1 | 79.6 +/- 3.9 | 35.2 +/- 2.4 | 410 |
| 200 | 62.0 +/- 5.1 | 90.2 +/- 3.4 | 34.0 +/- 3.7 | 39.4 +/- 2.6 | 85.8 +/- 2.2 | 38.4 +/- 2.4 | 430 |

## Qualitative checks against Shiu et al. 2024

- [x] contralateral MN9 fires at 100 Hz sugar drive
- [x] contralateral MN9 > ipsilateral MN9 at every rate >= 50 Hz
- [x] MN9 rate increases with GRN rate (Spearman > 0.9)
- [x] MN9 at 100 Hz is ~80% of its maximum (paper: w_syn tuned for this; accept 50-100%)
- [x] MN6 and MN8 also respond at 200 Hz
- [x] responders at max rate within 200-800 (paper: 455 at 200 Hz, v630)

**Result: PASS**

## Top 25 responders at 200 Hz

| root_id | rate (Hz) | super_class | type | side | community label |
|---|---|---|---|---|---|
| 720575940638202345 | 203.4 | sensory | LB3 | left | sensory; putative labial GRN; Sensory |
| 720575940629176663 | 202.6 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Putative Sugar Gustatory Receptor Neur |
| 720575940621502051 | 202.6 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Putative Sugar Gustatory Receptor Neur |
| 720575940630233916 | 202.4 | sensory | LB3 | left | gustatory receptor neuron/GRN?; putative labial GRN; Putative Sugar Gu |
| 720575940611875570 | 202.4 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Sensory |
| 720575940617000768 | 201.6 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Putative Sugar Gustatory Receptor Neur |
| 720575940617937543 | 201.0 | sensory | LB3 | left | gustatory receptor neuron/GRN? |
| 720575940632425919 | 199.8 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Putative Sugar Gustatory Receptor Neur |
| 720575940639332736 | 199.6 | sensory | LB3 | left | sensory; Putative Sugar Gustatory Receptor Neuron (GRN); putative GRN |
| 720575940616885538 | 199.2 | sensory | LB3 | left | gustatory receptor neuron/GRN?; putative labial GRN; Putative Sugar Gu |
| 720575940612670570 | 199.0 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Possible Sugar Gustatory Receptor Neur |
| 720575940630797113 | 199.0 | sensory | LB3 | left | Putative Sugar Gustatory Receptor Neuron (GRN) |
| 720575940624963786 | 198.8 | sensory | LB3 | left | Putative Sugar Gustatory Receptor Neuron (GRN) |
| 720575940632889389 | 196.6 | sensory | LB3 | left | gustatory receptor neuron/GRN?; sensory; Sensory |
| 720575940639198653 | 195.0 | sensory | LB3 | left | Sensory |
| 720575940633143833 | 194.4 | sensory | LB3 | left | Possible Sugar Gustatory Receptor Neuron (GRN) |
| 720575940621754367 | 194.0 | sensory | LB3 | left | sensory; putative labial GRN; Putative Sugar Gustatory Receptor Neuron |
| 720575940640649691 | 193.0 | sensory | LB3 | left | sensory; Putative Sugar Gustatory Receptor Neuron (GRN) |
| 720575940628853239 | 191.8 | sensory | LB3 | left | sensory; Possible Sugar Gustatory Receptor Neuron (GRN) |
| 720575940637568838 | 183.0 | sensory | LB3 | left | gustatory receptor neuron/GRN?; Putative Sugar Gustatory Receptor Neur |
| 720575940622695448 | 150.0 | central | CB0248 | right | unclassified_IN_FW_628_left; Specter (Shiu, Sterne et al., 2022 |
| 720575940629888530 | 137.8 | central | CB0192 | left | Second-order gustatory neuron Zorro (Shiu, Sterne et al., 2022 |
| 720575940627383685 | 137.0 | central | CB0248 | left | unclassified_IN_FW_556; Second-order gustatory neuron Billiards (Shiu, |
| 720575940618165019 | 124.2 | motor | CB0700 | left | Proboscis motor neuron/MN in pharyngeal nerve |
| 720575940630868793 | 121.2 | motor | CB0700 | right | Proboscis motor neuron/MN in pharyngeal nerve |

## Comparison with the authors' Brian2 model on the same v783 tables

Reference: Shiu et al. `model.py` run unmodified (Brian2, 2 trials x 1 s). Per-neuron mean rates over the union of responders.

| GRN rate | MN9 contra ours | MN9 contra Brian2 | responders ours | responders Brian2 | Pearson r (per-neuron rates) | median abs diff (Hz) |
|---|---|---|---|---|---|---|
| 50 | 10.2 | 14.5 | 289 | 276 | 0.987 | 1.2 |
| 100 | 62.8 | 62.0 | 374 | 366 | 0.994 | 1.1 |
| 150 | 79.2 | 88.5 | 393 | 385 | 0.996 | 1.6 |
| 200 | 90.2 | 90.0 | 430 | 415 | 0.998 | 0.9 |
