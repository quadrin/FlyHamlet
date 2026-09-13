#!/usr/bin/env python3
"""Run the reference Shiu et al. 2024 Brian2 model on the v783 tables.

Serves two purposes: a wall-clock baseline for the benchmark, and reference
MN9 firing rates for validation. Run inside the Brian2 venv:

    /home/user/venv-brian2/bin/python scripts/benchmark_brian2.py <shiu_repo> <out_dir> [rates...]
"""
import sys, time, json
from pathlib import Path
repo = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rates = [int(r) for r in sys.argv[3:]] or [100, 200]
sys.path.insert(0, str(repo))
import brian2
from brian2 import Hz, prefs
prefs.codegen.target = 'cython'
from model import run_exp, default_params
import utils as utl
import pandas as pd

neu_sugar = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,720575940617000768,
 720575940630797113,720575940632889389,720575940621754367,720575940621502051,720575940640649691,
 720575940639332736,720575940616885538,720575940639198653,720575940617937543,
 720575940632425919,720575940633143833,720575940612670570,720575940628853239,720575940629176663,720575940611875570]
ids_mn9 = [720575940660219265, 720575940645521262]
params = dict(default_params); params['n_run'] = 2
config = {'path_res': str(out), 'path_comp': str(repo/'Completeness_783.csv'), 'path_con': str(repo/'Connectivity_783.parquet'), 'n_proc': 1}
timing = {}
for r in rates:
    params['r_poi'] = r * Hz
    t0 = time.time()
    run_exp(exp_name=f'sugarR_{r}Hz', neu_exc=neu_sugar, params=params, **config, force_overwrite=True)
    timing[r] = time.time() - t0
    print(f'rate {r} Hz: {timing[r]:.0f} s wall for {params["n_run"]} trials of 1 s', flush=True)
ps = [str(out / f'sugarR_{r}Hz.parquet') for r in rates]
df = utl.load_exps(ps)
df_rate, df_std = utl.get_rate(df, t_run=1.0, n_run=params['n_run'])
df_rate.fillna(0).to_csv(out / 'brian2_rates.csv'); df_std.fillna(0).to_csv(out / 'brian2_rates_std.csv')
print('MN9 rates (Hz):'); print(df_rate.reindex(ids_mn9).fillna(0))
print('n active neurons per condition:', (df_rate.fillna(0) > 0).sum().to_dict())
print('top 15 at max rate:'); print(df_rate.sort_values(df_rate.columns[-1], ascending=False).head(15))
json.dump({'timing_s_per_2_trials': timing, 'brian2': brian2.__version__}, open(out / 'brian2_timing.json', 'w'), indent=1)
