"""Phase 1 validation: sugar GRN activation must drive the feeding motor neurons.

Reproduces Shiu et al. 2024 Fig. 1: labellar sugar GRNs (one hemisphere) are
activated with Poisson input at 10-200 Hz; the contralateral proboscis motor
neuron MN9 (and MN6, MN8) should fire, increasing with drive, and a few
hundred neurons should respond (45 at 10 Hz, 455 at 200 Hz in the paper's v630 model).

If ``results/benchmark/brian2/brian2_rates.csv`` exists (reference Brian2 run of the
authors' own code on the v783 tables), per-neuron rates are compared directly.

    python -m flyhamlet.validate [--trials 5] [--procs 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import load_connectome
from .sim import LIFNetwork

_C = None
_CFG = None


def _trial(args):
    rate, trial = args
    seed = int(_CFG["validation"]["seed"]) * 1000 + int(rate) * 10 + trial
    net = LIFNetwork(_C, _CFG["sim"], seed=seed, backend="numba")
    present = set(_C.root_ids.tolist())
    ids = [i for i in _CFG["validation"]["sugar_grn_ids"] if i in present]
    net.inject_poisson({"ids": ids}, float(rate))
    rec = net.run(duration_s=float(_CFG["validation"]["t_run_s"]))
    return rate, trial, rec.rates_hz(_C.n).astype(np.float32)


def main(argv=None):
    global _C, _CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args(argv)
    _CFG = load_config(a.config)
    _C = load_connectome(_CFG, verbose=False)
    c, cfg = _C, _CFG
    val = cfg["validation"]
    rates = list(val["rates_hz"])
    n_tr = a.trials or int(val["n_trials"])
    out = resolve(cfg, a.out); out.mkdir(parents=True, exist_ok=True)

    present = set(c.root_ids.tolist())
    sugar_idx = c.index_of([i for i in val["sugar_grn_ids"] if i in present])
    grn_side = pd.Series(c.neurons["side"].to_numpy()[sugar_idx]).mode()[0]
    mn9 = c.index_of(val["mn9_ids"])
    mn = {"MN9": mn9, "MN6": c.neurons_with_label(r"Motor neuron 6; MN6"), "MN8": c.neurons_with_label(r"Motor neuron 8; MN8")}
    side = c.neurons["side"].to_numpy()

    jobs = [(r, t) for r in rates for t in range(n_tr)]
    print(f"validation: {len(sugar_idx)} sugar GRNs (side={grn_side}), rates {rates} Hz x {n_tr} trials, {a.procs} procs", file=sys.stderr)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs) as pool:
        res = pool.map(_trial, jobs, chunksize=1)
    R = {r: np.stack([x[2] for x in res if x[0] == r]) for r in rates}     # rate -> (trials, N)
    mean = pd.DataFrame({r: R[r].mean(0) for r in rates}); std = pd.DataFrame({r: R[r].std(0) for r in rates})
    mean.index = c.root_ids; std.index = c.root_ids
    mean.to_csv(out / "rates_mean.csv"); std.to_csv(out / "rates_std.csv")

    # ---- report
    L = ["# Phase 1 validation: sugar GRN -> feeding motor neurons", "",
         f"{len(sugar_idx)} labellar sugar GRNs (v783 side = {grn_side}) driven by Poisson input; {n_tr} trials x {val['t_run_s']} s per rate; "
         f"mean +/- s.d. across trials. Shiu et al. 2024 parameters (w_syn = {cfg['sim']['w_syn_mV']} mV).", "",
         "## Motor neuron firing rates (Hz)", "", "| GRN rate | " + " | ".join(
             f"{k} {'contra' if side[i] != grn_side else 'ipsi'}" for k, ix in mn.items() for i in ix) + " | responders (>0 Hz) |",
         "|---|" + "---|" * (sum(len(ix) for ix in mn.values()) + 1)]
    for r in rates:
        cells = [f"{mean[r].iloc[i]:.1f} +/- {std[r].iloc[i]:.1f}" for ix in mn.values() for i in ix]
        L.append(f"| {r} | " + " | ".join(cells) + f" | {int((mean[r] > 0).sum())} |")
    mn9_contra = [i for i in mn9 if side[i] != grn_side][0]; mn9_ipsi = [i for i in mn9 if side[i] == grn_side][0]
    r100 = mean[100].iloc[mn9_contra] if 100 in rates else float("nan")
    rmax = max(mean[r].iloc[mn9_contra] for r in rates)
    checks = [
        ("contralateral MN9 fires at 100 Hz sugar drive", r100 > 0),
        ("contralateral MN9 > ipsilateral MN9 at every rate >= 50 Hz", all(mean[r].iloc[mn9_contra] >= mean[r].iloc[mn9_ipsi] for r in rates if r >= 50)),
        ("MN9 rate increases with GRN rate (Spearman > 0.9)", pd.Series([mean[r].iloc[mn9_contra] for r in rates]).corr(pd.Series(rates), method="spearman") > 0.9),
        ("MN9 at 100 Hz is ~80% of its maximum (paper: w_syn tuned for this; accept 50-100%)", 0.5 <= r100 / max(rmax, 1e-9) <= 1.0),
        ("MN6 and MN8 also respond at 200 Hz", all(mean[max(rates)].iloc[i] > 0 for k in ("MN6", "MN8") for i in mn[k])),
        ("responders at max rate within 200-800 (paper: 455 at 200 Hz, v630)", 200 <= int((mean[max(rates)] > 0).sum()) <= 800),
    ]
    L += ["", "## Qualitative checks against Shiu et al. 2024", ""] + [f"- [{'x' if ok else ' '}] {name}" for name, ok in checks]
    ok_all = all(ok for _, ok in checks)
    L += ["", f"**Result: {'PASS' if ok_all else 'FAIL'}**"]

    # ---- top responders
    lab = c.labels.groupby("root_id")["label"].apply(lambda s: "; ".join(dict.fromkeys(x.split(";")[0].strip() for x in s))[:70])
    top = mean[max(rates)].sort_values(ascending=False).head(25)
    L += ["", f"## Top 25 responders at {max(rates)} Hz", "", "| root_id | rate (Hz) | super_class | type | side | community label |", "|---|---|---|---|---|---|"]
    for rid, v in top.items():
        i = c.index_of(rid)[0]; n = c.neurons.iloc[i]
        L.append(f"| {rid} | {v:.1f} | {n['super_class']} | {n['consolidated_type']} | {n['side']} | {lab.get(rid, '')} |")

    # ---- Brian2 reference comparison
    ref_p = resolve(cfg, "results/benchmark/brian2/brian2_rates.csv")
    if ref_p.exists():
        ref = pd.read_csv(ref_p, index_col=0)
        ref.columns = [int(cc.replace("sugarR_", "").replace("Hz", "")) for cc in ref.columns]
        L += ["", "## Comparison with the authors' Brian2 model on the same v783 tables", "",
              "Reference: Shiu et al. `model.py` run unmodified (Brian2, 2 trials x 1 s). Per-neuron mean rates over the union of responders.", "",
              "| GRN rate | MN9 contra ours | MN9 contra Brian2 | responders ours | responders Brian2 | Pearson r (per-neuron rates) | median abs diff (Hz) |", "|---|---|---|---|---|---|---|"]
        for r in sorted(set(ref.columns) & set(rates)):
            ours = mean[r]; theirs = ref[r].reindex(ours.index).fillna(0)
            m = (ours > 0) | (theirs > 0)
            rr = np.corrcoef(ours[m], theirs[m])[0, 1] if m.sum() > 2 else float("nan")
            L.append(f"| {r} | {ours.iloc[mn9_contra]:.1f} | {theirs.iloc[mn9_contra]:.1f} | {int((ours > 0).sum())} | {int((theirs > 0).sum())} | {rr:.3f} | {np.median(np.abs(ours[m] - theirs[m])):.1f} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, ix in mn.items():
            for i in ix:
                lbl = f"{k} {'contra' if side[i] != grn_side else 'ipsi'}"
                ax[0].errorbar(rates, [mean[r].iloc[i] for r in rates], [std[r].iloc[i] for r in rates], marker="o", label=lbl, capsize=2)
        if ref_p.exists():
            rr = sorted(set(ref.columns) & set(rates))
            ax[0].plot(rr, [ref[r].reindex([c.root_ids[mn9_contra]]).fillna(0).iloc[0] for r in rr], "k--", marker="s", label="MN9 contra, Brian2 ref")
        ax[0].set_xlabel("sugar GRN rate (Hz)"); ax[0].set_ylabel("motor neuron rate (Hz)"); ax[0].legend(fontsize=7); ax[0].set_title("Feeding motor neurons")
        ax[1].plot(rates, [(mean[r] > 0).sum() for r in rates], marker="o"); ax[1].set_xlabel("sugar GRN rate (Hz)"); ax[1].set_ylabel("neurons with rate > 0")
        ax[1].set_title("Responders")
        fig.tight_layout(); fig.savefig(out / "validation.png", dpi=130)
    except Exception as e:  # pragma: no cover
        print("figure failed:", e, file=sys.stderr)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
