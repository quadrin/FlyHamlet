# Backend benchmark

Full v783 connectome (139,255 neurons, 15.09M connections), dt = 0.1 ms, 20 sugar GRNs driven at 150 Hz, 1.0 s simulated, single process.

| backend | wall (s) | ms / step | x real time | spikes | active neurons | MN9 L/R (Hz) | identical spikes |
|---|---|---|---|---|---|---|---|
| numba | 3.3 | 0.329 | 3.3x | 15647 | 377 | 59/98 | yes |
| numpy | 11.2 | 1.125 | 11.2x | 15647 | 377 | 59/98 | yes |
| torch | 13.5 | 1.352 | 13.5x | 15647 | 377 | 59/98 | yes |

Brian2 2.9.0 reference (Shiu et al. model.py, cython codegen, 1 process, includes network build per trial):

- 50 Hz drive: 86 s per 1 s trial (86x real time)
- 100 Hz drive: 69 s per 1 s trial (69x real time)
- 150 Hz drive: 66 s per 1 s trial (66x real time)
- 200 Hz drive: 71 s per 1 s trial (71x real time)

No GPU was available in this environment; the torch backend was timed on CPU only.
Run one backend per process: numba's and torch's thread pools contend badly when mixed in one process.
