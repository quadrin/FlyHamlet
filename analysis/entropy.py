#!/usr/bin/env python3
"""Experiment 1: typing entropy of connectome-driven flies.

Reads keystroke logs (``fly*_keystrokes.csv``) and reports keystroke statistics,
marginal / conditional / Lempel-Ziv entropies, transition-graph structure, two
null models and the probability of typing Hamlet under the fitted first-order
model.  Writes ``report.md`` plus PNGs (transition heatmap, key-visit heatmap,
sample trajectory).

    python analysis/entropy.py --logs 'results/arena/fly*_keystrokes.csv' --out results/entropy_exp1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flyhamlet.config import load_config, resolve  # noqa: E402

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
K = len(ALPHABET)
IDX = {ch: i for i, ch in enumerate(ALPHABET)}
LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- estimators
def encode(text: str) -> np.ndarray:
    return np.fromiter((IDX[ch] for ch in text if ch in IDX), dtype=np.int64)


def plugin_entropy_bits(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def miller_madow_bits(counts: np.ndarray) -> float:
    """Plug-in entropy plus the Miller-Madow bias correction (m-1)/(2N) nats."""
    n = counts.sum()
    m = int((counts > 0).sum())
    if n == 0 or m <= 1:
        return 0.0
    return plugin_entropy_bits(counts) + (m - 1) / (2 * n) / LOG2


def transition_counts(seq: np.ndarray, k: int = K) -> np.ndarray:
    """k x k count matrix of consecutive pairs within one sequence."""
    T = np.zeros((k, k), dtype=np.int64)
    if len(seq) > 1:
        np.add.at(T, (seq[:-1], seq[1:]), 1)
    return T


def conditional_entropy_bits(T: np.ndarray, correction: str = "plugin") -> float:
    """H(X_{t+1} | X_t) = sum_i p(i) H(row i), rows weighted by their out-counts."""
    n = T.sum()
    if n == 0:
        return 0.0
    h = 0.0
    for i in range(T.shape[0]):
        r = T[i].sum()
        if r == 0:
            continue
        hi = miller_madow_bits(T[i]) if correction == "miller_madow" else plugin_entropy_bits(T[i])
        h += r / n * hi
    return float(h)


def lz76_complexity(seq) -> int:
    """Number of phrases in the Lempel-Ziv 1976 parsing (Kaspar & Schuster 1987 counting).

    Each phrase is the shortest string starting at the current position that does not
    occur (as a substring, overlaps allowed) in the text seen so far; a final incomplete
    phrase counts as one.  Linear time: the phrase is matched by walking an online suffix
    automaton that always indexes exactly the text before the character being matched.
    """
    from analysis.typist import SuffixAutomaton
    s = [chr(int(x)) for x in seq] if not isinstance(seq, str) else list(seq)
    n = len(s)
    if n == 0:
        return 0
    sam = SuffixAutomaton()
    nxt, link, length = sam.next, sam.link, sam.length
    state, l, c, j = 0, 0, 0, 0          # walker state, matched phrase length, phrase count, chars indexed
    for k in range(n):
        while j < k:                     # index s[0:k] so a match of s[i:k+1] must start before i
            sam.extend(s[j]); j += 1
        while state != 0 and length[link[state]] >= l:   # re-home the walker after clone splits
            state = link[state]
        t = nxt[state].get(s[k])
        if t is not None:
            state, l = t, l + 1
        else:
            c += 1; state, l = 0, 0
    return c + (1 if l > 0 else 0)


def lz_entropy_rate_bits(seq, alphabet_size: int = K) -> float:
    """LZ76 entropy-rate estimate  h = c(n) * log2(n) / n  (bits/symbol).

    Converges to the entropy rate slowly from above; for short sequences it is an
    upper-biased estimate and is reported alongside the null models measured
    with the same estimator on the same length.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    return lz76_complexity(seq) * math.log2(n) / n


def graph_structure(T: np.ndarray, visits: np.ndarray) -> dict:
    """Zero-visit keys, strong connectivity of the observed transition graph, absorbing keys."""
    zero = [ALPHABET[i] for i in np.flatnonzero(visits == 0)]
    A = sp.csr_matrix((T > 0).astype(np.int8))
    visited = np.flatnonzero(visits > 0)
    if len(visited):
        sub = A[visited][:, visited]
        ncomp, labels = connected_components(sub, directed=True, connection="strong")
    else:
        ncomp = 0
    absorbing = []
    for i in visited:
        out = T[i].copy(); out[i] = 0
        if out.sum() == 0:
            absorbing.append(ALPHABET[i])
    return dict(zero_visit_keys=zero, n_visited=int(len(visited)), n_strong_components=int(ncomp),
                single_scc=bool(ncomp == 1 and len(visited) > 0), absorbing_keys=absorbing)


# --------------------------------------------------------------------------- null models
def grid_random_walk(rows: int, cols: int, n_steps: int, seed: int, letters: list[str] | None = None) -> np.ndarray:
    """Uniform random walk on the rows x cols grid (4-neighbour moves, off-grid moves rejected)."""
    rng = np.random.default_rng(seed)
    r, c = rows // 2, cols // 2
    out = np.empty(n_steps, dtype=np.int64)
    moves = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
    draws = rng.integers(0, 4, size=4 * n_steps)
    j = 0
    for i in range(n_steps):
        while True:
            dr, dc = moves[draws[j]]; j += 1
            if j >= len(draws):
                draws = rng.integers(0, 4, size=4 * n_steps); j = 0
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                r, c = nr, nc
                break
        out[i] = r * cols + c
    if letters is not None:   # map key index -> alphabet index
        lut = np.array([IDX[l] for l in letters])
        out = lut[out]
    return out


# --------------------------------------------------------------------------- Hamlet
def fetch_hamlet(url: str, cache: Path) -> str:
    cache = Path(cache)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            raw = r.read().decode("utf-8", errors="replace")
        cache.write_text(raw)
    raw = cache.read_text()
    m = re.search(r"\*\*\* START OF [^\n]*\*\*\*(.*)\*\*\* END OF", raw, re.S)
    body = m.group(1) if m else raw
    return clean_text(body)


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z ]+", " ", s.replace("\n", " "))
    return re.sub(r" +", " ", s).strip()


def log2_prob_under_model(text: str, p0: np.ndarray, P: np.ndarray) -> tuple[float, list, int]:
    """log2 P(text) under a first-order model (marginal p0 for the first symbol, transition matrix P).

    Returns (log2 prob or -inf, list of zero-probability (pair, frequency-in-text) items, n_transitions).
    """
    seq = encode(text)
    T = transition_counts(seq)
    zero = []
    lp = 0.0
    if p0[seq[0]] <= 0:
        zero.append((ALPHABET[seq[0]] + "(start)", 1))
    else:
        lp += math.log2(p0[seq[0]])
    with np.errstate(divide="ignore"):
        L = np.log2(P)
    mask = T > 0
    bad = mask & (P <= 0)
    for i, j in zip(*np.nonzero(bad)):
        zero.append((ALPHABET[i] + ALPHABET[j], int(T[i, j])))
    if zero:
        return -math.inf, sorted(zero, key=lambda x: -x[1]), int(T.sum())
    lp += float((T[mask] * L[mask]).sum())
    return lp, [], int(T.sum())


# --------------------------------------------------------------------------- report
def analyse(seqs: list[np.ndarray], durations: list[float], letters: list[str] | None, hamlet: str | None,
            null_steps: int, null_seed: int, rows: int, cols: int) -> dict:
    allseq = np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.int64)
    n = len(allseq)
    visits = np.bincount(allseq, minlength=K)
    T = sum((transition_counts(s) for s in seqs), np.zeros((K, K), dtype=np.int64))
    res = dict(n_keystrokes=int(n), total_time_s=float(sum(durations)), rows=rows, cols=cols,
               rate_per_s=float(n / sum(durations)) if sum(durations) > 0 else float("nan"),
               H0_plugin=plugin_entropy_bits(visits), H0_miller_madow=miller_madow_bits(visits),
               H_cond_plugin=conditional_entropy_bits(T), H_cond_miller_madow=conditional_entropy_bits(T, "miller_madow"),
               H_lz=lz_entropy_rate_bits(allseq), visits=visits, T=T, graph=graph_structure(T, visits), n_flies=len(seqs))
    # null 1: uniform random walk on the grid, same estimators on 1e6 steps and on a length-matched prefix
    walk = grid_random_walk(rows, cols, null_steps, null_seed, letters)
    Tw = transition_counts(walk); vw = np.bincount(walk, minlength=K)
    res["null_walk"] = dict(H0=miller_madow_bits(vw), H_cond=conditional_entropy_bits(Tw), H_lz=lz_entropy_rate_bits(walk[:200000]),
                            H_lz_matched=lz_entropy_rate_bits(walk[:max(2, n)]), n=null_steps)
    # exact conditional entropy of the 4-neighbour walk: sum over cells p(cell) log2(#neighbours)
    deg = np.array([[sum(1 for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= r + dr < rows and 0 <= c + dc < cols)
                     for c in range(cols)] for r in range(rows)], dtype=float).ravel()
    pi = deg / deg.sum()   # stationary distribution of the simple random walk is proportional to degree
    res["null_walk"]["H_cond_exact"] = float((pi * np.log2(deg)).sum())
    res["null_walk"]["H0_exact"] = float(-(pi * np.log2(pi)).sum())
    # null 2: uniform i.i.d. typist
    res["null_uniform"] = dict(H0=math.log2(K), H_cond=math.log2(K),
                               H_lz_matched=lz_entropy_rate_bits(np.random.default_rng(null_seed + 1).integers(0, K, max(2, n))))
    # Hamlet
    if hamlet:
        p0 = visits / max(1, visits.sum())
        rows_sum = T.sum(1, keepdims=True)
        P = np.divide(T, rows_sum, out=np.zeros_like(T, dtype=float), where=rows_sum > 0)
        lp, zero, ntr = log2_prob_under_model(hamlet, p0, P)
        hres = dict(len=len(hamlet), n_transitions=ntr, log2_prob=lp, zero_pairs=zero,
                    log10_27_pow_len=len(hamlet) * math.log10(K))
        if letters is not None:   # geometry: which Hamlet pairs are even possible on this key grid?
            pos = {IDX[l]: divmod(k, cols) for k, l in enumerate(letters)}
            Th = transition_counts(encode(hamlet))
            need = [(i, j) for i, j in zip(*np.nonzero(Th))]
            adj = [(i, j) for i, j in need if i != j and max(abs(pos[i][0] - pos[j][0]), abs(pos[i][1] - pos[j][1])) <= 1]
            hres["hamlet_distinct_pairs"] = len(need)
            hres["hamlet_pairs_grid_adjacent"] = len(adj)
            hres["hamlet_transitions_grid_adjacent"] = int(sum(Th[i, j] for i, j in adj))
        if zero:
            hres["n_zero_pairs"] = len(zero)
            hres["hamlet_transitions_missing"] = int(sum(f for _, f in zero))
        else:
            hres["log10_expected_keystrokes"] = -lp * math.log10(2)
        # Laplace-smoothed variant (pseudo-count 0.5) so a number exists even with unseen pairs
        Ps = (T + 0.5) / (T + 0.5).sum(1, keepdims=True)
        p0s = (visits + 0.5) / (visits + 0.5).sum()
        lps, _, _ = log2_prob_under_model(hamlet, p0s, Ps)
        hres["log2_prob_smoothed"] = lps
        hres["log10_expected_keystrokes_smoothed"] = -lps * math.log10(2)
        res["hamlet"] = hres
    return res


def write_report(res: dict, out: Path, letters: list[str] | None, traj_file: Path | None):
    g = res["graph"]
    L = ["# Experiment 1: typing entropy", "",
         f"{res['n_flies']} flies, {res['n_keystrokes']} keystrokes in {res['total_time_s']:.0f} s of simulated time "
         f"({res['rate_per_s']:.3f} keystrokes/s).", "",
         "## Entropies (bits per keystroke)", "",
         "| quantity | fly typist | uniform grid walk (null, 1e6 steps) | uniform 27-key typist (null) |", "|---|---|---|---|",
         f"| H0 marginal, plug-in | {res['H0_plugin']:.3f} | {res['null_walk']['H0_exact']:.3f} (exact) | {math.log2(K):.3f} |",
         f"| H0 marginal, Miller-Madow | {res['H0_miller_madow']:.3f} | {res['null_walk']['H0']:.3f} | {math.log2(K):.3f} |",
         f"| H(X_t+1 \\| X_t), plug-in | {res['H_cond_plugin']:.3f} | {res['null_walk']['H_cond']:.3f} (exact {res['null_walk']['H_cond_exact']:.3f}) | {math.log2(K):.3f} |",
         f"| H(X_t+1 \\| X_t), Miller-Madow rows | {res['H_cond_miller_madow']:.3f} | | |",
         f"| LZ76 entropy rate (this length) | {res['H_lz']:.3f} | {res['null_walk']['H_lz_matched']:.3f} | {res['null_uniform']['H_lz_matched']:.3f} |",
         f"| LZ76 entropy rate (2e5 symbols) | | {res['null_walk']['H_lz']:.3f} | |", "",
         "The LZ76 estimate is upper-biased at short lengths, so the null models are also evaluated on sequences of the same length.", "",
         "## Transition graph", "",
         f"- keys visited: {g['n_visited']} of {K}; zero-visit keys: {g['zero_visit_keys'] or 'none'}",
         f"- strongly connected components among visited keys: {g['n_strong_components']} "
         f"({'one SCC: every visited key can reach every other' if g['single_scc'] else 'NOT a single SCC'})",
         f"- absorbing keys (visited, no exit observed): {g['absorbing_keys'] or 'none'}", ""]
    if "hamlet" in res:
        h = res["hamlet"]
        L += ["## Hamlet", "", f"Hamlet (Project Gutenberg #1524, lowercase a-z + space, whitespace collapsed): {h['len']:,} characters, "
              f"{h['n_transitions']:,} transitions. 27^len = 10^{h['log10_27_pow_len']:.0f}.", ""]
        if "hamlet_pairs_grid_adjacent" in h:
            L += [f"Geometry: a keystroke is logged only on entering a new key region, so consecutive keys must be neighbours on the "
                  f"{res.get('rows', 3)}x{res.get('cols', 9)} grid (8-neighbourhood, corner crossings included). Of Hamlet's "
                  f"{h['hamlet_distinct_pairs']} distinct letter pairs only {h['hamlet_pairs_grid_adjacent']} are grid-adjacent under this layout "
                  f"({h['hamlet_transitions_grid_adjacent']:,} of {h['n_transitions']:,} transitions). Every other pair has probability zero for "
                  f"any fly, however it moves: on a grid typewriter Hamlet is unreachable unless the layout makes all of its pairs adjacent.", ""]
        if h["zero_pairs"]:
            L += [f"Under the fitted first-order model P(Hamlet) = 0: {h['n_zero_pairs']} letter pairs needed by Hamlet were never typed "
                  f"({h['hamlet_transitions_missing']:,} of Hamlet's transitions). Missing pairs and their frequencies in Hamlet "
                  f"(top 40):", "", "| pair | count in Hamlet |", "|---|---|"]
            L += [f"| `{p}` | {f:,} |" for p, f in h["zero_pairs"][:40]]
            L += ["", f"With Laplace smoothing (pseudo-count 0.5): log2 P = {h['log2_prob_smoothed']:.0f}, expected keystrokes ~ 10^{h['log10_expected_keystrokes_smoothed']:.0f}."]
        else:
            L += [f"log2 P(Hamlet) = {h['log2_prob']:.0f} under the fitted model, i.e. expected keystrokes-to-Hamlet ~ 10^{h['log10_expected_keystrokes']:.0f} "
                  f"(compare 27^len = 10^{h['log10_27_pow_len']:.0f}; smoothed model: 10^{h['log10_expected_keystrokes_smoothed']:.0f})."]
        L += [""]
    L += ["## Figures", "", "![transition heatmap](transition_heatmap.png)", "", "![key visits](key_visit_heatmap.png)", ""]
    if traj_file is not None:
        L += ["![trajectory](trajectory.png)", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}, open(out / "results.json", "w"), indent=1, default=str)


def make_figures(res: dict, out: Path, letters: list[str] | None, rows: int, cols: int, traj_file: Path | None,
                 arena_wh: tuple[float, float] | None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ink, muted, grid = "#1f2328", "#6e7781", "#d0d7de"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted})
    T = res["T"]; rows_sum = T.sum(1, keepdims=True)
    P = np.divide(T, rows_sum, out=np.zeros_like(T, dtype=float), where=rows_sum > 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=max(1e-9, P.max()))
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([repr(c) if c == " " else c for c in ALPHABET]); ax.set_yticklabels([repr(c) if c == " " else c for c in ALPHABET])
    ax.set_xlabel("next key"); ax.set_ylabel("current key"); ax.set_title("Transition probabilities P(next | current)", loc="left", color=ink)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="probability")
    fig.tight_layout(); fig.savefig(out / "transition_heatmap.png", dpi=140); plt.close(fig)

    if letters is not None:
        V = np.array([res["visits"][IDX[l]] for l in letters]).reshape(rows, cols)
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        im = ax.imshow(V, cmap="Blues", origin="lower", vmin=0)
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, f"{letters[r * cols + c]!r}\n{int(V[r, c])}", ha="center", va="center", fontsize=8,
                        color="white" if V[r, c] > 0.6 * max(1, V.max()) else ink)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Key visits (entries per key region)", loc="left", color=ink)
        for s in ax.spines.values(): s.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="entries")
        fig.tight_layout(); fig.savefig(out / "key_visit_heatmap.png", dpi=140); plt.close(fig)

    if traj_file is not None and traj_file.exists() and arena_wh is not None:
        df = pd.read_csv(traj_file); W, H = arena_wh
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(plt.Rectangle((c * W / cols, r * H / rows), W / cols, H / rows, fill=False, lw=0.6, color=grid))
                if letters is not None:
                    ax.text((c + 0.5) * W / cols, (r + 0.5) * H / rows, repr(letters[r * cols + c]), ha="center", va="center", color=muted, fontsize=8)
        sc = ax.scatter(df["x"], df["y"], c=df["t_s"], s=2, cmap="viridis", lw=0)
        ax.plot(df["x"].iloc[0], df["y"].iloc[0], "o", color="#cf222e", ms=5, label="start")
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title(f"Sample trajectory ({traj_file.stem.replace('_trajectory', '')}), colour = time", loc="left", color=ink)
        fig.colorbar(sc, ax=ax, shrink=0.8, label="time (s)")
        for s in ax.spines.values(): s.set_visible(False)
        fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=140); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--logs", default=None, help="glob of keystroke CSVs (default: <arena.log_dir>/fly*_keystrokes.csv)")
    ap.add_argument("--out", default="results/entropy_exp1")
    ap.add_argument("--no-hamlet", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    logs = sorted(glob.glob(a.logs or str(resolve(cfg, cfg["arena"]["log_dir"]) / "fly*_keystrokes.csv")))
    if not logs:
        sys.exit("no keystroke logs found")
    out = resolve(cfg, a.out); out.mkdir(parents=True, exist_ok=True)
    seqs, durations, letters = [], [], None
    for f in logs:
        df = pd.read_csv(f, dtype={"letter": str}, keep_default_na=False)
        seqs.append(encode("".join(df["letter"].tolist())))
        summ = Path(f).with_name(Path(f).name.replace("_keystrokes.csv", "_summary.json"))
        if summ.exists():
            s = json.load(open(summ)); durations.append(float(s["sim_time_s"])); letters = letters or s.get("layout")
        else:
            durations.append(float(df["sim_time_s"].max()) if len(df) else 0.0)
    tw = cfg["arena"]["typewriter"]
    hamlet = None if a.no_hamlet else fetch_hamlet(cfg["analysis"]["hamlet_url"], resolve(cfg, cfg["analysis"]["hamlet_cache"]))
    res = analyse(seqs, durations, letters, hamlet, int(cfg["analysis"]["null_walk_steps"]), int(cfg["analysis"]["null_seed"]),
                  int(tw["rows"]), int(tw["cols"]))
    traj = Path(logs[0]).with_name(Path(logs[0]).name.replace("_keystrokes.csv", "_trajectory.csv"))
    write_report(res, out, letters, traj if traj.exists() else None)
    make_figures(res, out, letters, int(tw["rows"]), int(tw["cols"]), traj, (float(cfg["arena"]["width_mm"]), float(cfg["arena"]["height_mm"])))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
