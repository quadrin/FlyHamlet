# Phase 2: choosing the motor readout and the looming input (v783 annotation check)

All names were looked up in the v783 tables (`classification_with_fw_and_hemibrain_types.csv.gz`
and `consolidated_cell_types.csv.gz`), not taken from memory.

| type | where it appears in v783 | n (left / right) | super class | nt |
|---|---|---|---|---|
| DNa02 | hemibrain_type, consolidated | 2 (1 / 1) | descending | ACh |
| DNa01 | cell_type, hemibrain_type, consolidated | 4 (2 / 2) | descending | ACh |
| DNp09 | hemibrain_type, consolidated | 4 (2 / 2) | descending | ACh |
| MDN | hemibrain_type, consolidated | 4 (2 / 2) | descending | ACh |
| LC4 | hemibrain_type, consolidated | 104 (54 / 50) | visual_projection | ACh |
| LPLC2 | hemibrain_type, consolidated | 210 (108 / 102) | visual_projection | ACh |
| Giant_Fiber (= DNp01) | hemibrain_type (`Giant_Fiber`), cell_type (`DNp01`) | 2 (1 / 1) | descending | ACh / GLUT |

## Screen: which descending neurons respond to unilateral looming input?

Poisson drive at 100 Hz on all LC4 + LPLC2 neurons of one eye for 1 s (seed 5), rates in Hz.

| drive | responders | GF L / R | DNa02 L / R | DNa01 L / R (mean) | DNp09 L / R | MDN L / R |
|---|---|---|---|---|---|---|
| left eye | 830 | 148 / 95 | 0 / 5 | 0 / 20 | 4 / 0 | 0 / 1.5 |
| right eye | 802 | 88 / 162 | 45 / 0 | 17.5 / 0 | 0 / 0.5 | 10 / 14 |

Other strongly lateralised responders: DNp103 (ipsilateral, 168 vs 18 Hz), DNp06 (contralateral, 69 Hz),
DNa13 (contralateral, 29-45 Hz), DNa03 (contralateral, 26 Hz).

## Decisions (all gains live in `config.yaml`)

* **Turning**: left-minus-right rate of **DNa02 + DNa01**. Both are the classic steering DNs
  (Rayshubskiy et al. 2020; Namiki et al. 2018) and both respond to *contralateral* looming in
  the connectome model, so a wall on the left activates the right-side DNs and the fly turns
  right, away from the wall. `turn_sign = +1` means excess activity on the left turns left
  (ipsilateral turning, as reported for DNa02).
* **Forward speed**: `base_speed` + `speed_gain` x mean **DNp09** rate - `backward_gain` x mean
  **MDN** rate. DNp09 is a forward-walking/freezing DN and responds only weakly here; MDN is the
  *moonwalker* descending neuron, which drives **backward** walking, so it is wired with a
  negative sign rather than as a forward-speed source. Without any input the whole network is
  silent, hence the non-zero `base_speed`.
* **Looming**: walls within `wall_distance_mm` along an eye's rays drive that eye's LC4 + LPLC2
  with a Poisson rate `max_rate * (1 - d / wall_distance)^exponent`. The giant fiber (DNp01) is
  the classic downstream target and fires strongly; nothing is read out from it for locomotion
  (the fly cannot take off), but its rate is logged in the trajectory file.

Observed behaviour: in a 10 s run the fly reached a wall region, looming input rose to
~70 Hz on the near eye, DNa02/DNa01 fired on the opposite side, the fly turned at up to
320 deg/s and left the wall without ever touching it (wall-contact fraction 0).
