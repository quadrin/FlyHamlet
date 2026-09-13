#!/usr/bin/env python3
"""Build the GitHub Pages viewer, learning lab, and saved trajectory files."""
import glob, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
COLS = ['t_s','x','y','heading_deg','v_mm_s','omega_deg_s','loom_L_hz','loom_R_hz','turnDN_L_hz','turnDN_R_hz','fwdDN_hz','bwdDN_hz','GF_hz']

def main(arena_dir="results/arena"):
    out = ROOT / "site/data"; out.mkdir(parents=True, exist_ok=True)
    n = 0
    for tf in sorted(glob.glob(str(ROOT / arena_dir / "fly*_trajectory.csv"))):
        # UI-only rebuilds reuse the committed recordings and need no pandas.
        import pandas as pd
        fid = Path(tf).name[3:5]
        tr = pd.read_csv(tf)
        ks = pd.read_csv(tf.replace("_trajectory.csv", "_keystrokes.csv"), keep_default_na=False)
        summ = json.load(open(tf.replace("_trajectory.csv", "_summary.json")))
        data = {c: [round(float(v), 2 if c in ('t_s','x','y') else 1) for v in tr[c]] for c in COLS}
        keys = [[round(float(r.sim_time_s), 2), int(r.key_index), r.letter] for r in ks.itertuples()]
        payload = dict(fly=fid, traj=data, keys=keys, layout=summ["layout"], seed=summ["seed"], duration=summ["sim_time_s"],
                       spikes=summ["spikes"], W=90, H=30, rows=3, cols=9)
        (out / f"fly{fid}.json").write_text(json.dumps(payload, separators=(",", ":")))
        n += 1
    (ROOT / "index.html").write_text("<!doctype html>\n<html lang=\"en\"><head>" + (ROOT / "site/head.html").read_text() + "</head><body>"
                                     + (ROOT / "site/body.html").read_text() + "</body></html>\n")
    (ROOT / "learn.html").write_text("<!doctype html>\n<html lang=\"en\"><head>" + (ROOT / "site/learn-head.html").read_text() + "</head><body>"
                                    + (ROOT / "site/learn-body.html").read_text() + "</body></html>\n")
    (ROOT / ".nojekyll").write_text("")
    print(f"wrote index.html, learn.html and {n} fly data files to site/data/")

if __name__ == "__main__":
    main(*sys.argv[1:])
