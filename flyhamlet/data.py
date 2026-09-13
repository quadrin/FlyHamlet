"""Phase 0: download the FlyWire v783 release tables and build the connectome.

The tables are the public Codex (codex.flywire.ai) exports for FlyWire FAFB
snapshot 783, served from the ``flywire-data`` Google Cloud Storage bucket.

The loader produces a :class:`Connectome` with

* a neuron index (``root_ids`` sorted, ``index_of`` for the reverse map),
* a sparse pre -> post synapse-count matrix (CSR, ``int32`` counts),
* per-neuron neurotransmitter and synaptic sign,
* per-neuron cell type (FlyWire type, hemibrain type, consolidated type),
  super class, class, sub class, side and nerve, plus community labels.

The processed result is cached as a single ``.npz`` + parquet so later phases
load in a second or two.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_config, resolve

NT_CODES = ["ACH", "GABA", "GLUT", "DA", "SER", "OCT", "UNK"]


# --------------------------------------------------------------------------- download
def download(cfg: dict, verbose: bool = True) -> dict[str, Path]:
    """Download every table listed in ``cfg['data']['files']`` that is not cached yet."""
    d = cfg["data"]
    raw = resolve(cfg, d["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fname in d["files"].items():
        dst = raw / fname
        if not dst.exists() or dst.stat().st_size == 0:
            url = f"{d['base_url'].rstrip('/')}/{fname}"
            if verbose:
                print(f"downloading {url} -> {dst}", file=sys.stderr)
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dst)
        out[key] = dst
    return out


# --------------------------------------------------------------------------- connectome
@dataclass
class Connectome:
    root_ids: np.ndarray                 # (N,) int64, sorted
    W: sp.csr_matrix                     # (N, N) int32 synapse counts, pre rows -> post cols
    nt: np.ndarray                       # (N,) str codes from NT_CODES
    sign: np.ndarray                     # (N,) int8: +1 excitatory, -1 inhibitory
    neurons: pd.DataFrame                # annotation table indexed 0..N-1
    labels: pd.DataFrame | None = None   # community labels (root_id, label)
    _index: dict = field(default_factory=dict, repr=False)

    # ----- basic maps
    @property
    def n(self) -> int:
        return len(self.root_ids)

    def index_of(self, ids: Iterable[int] | int, strict: bool = True) -> np.ndarray:
        """Map FlyWire root ids to dense indices. Missing ids raise (strict) or are dropped."""
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        pos = np.searchsorted(self.root_ids, ids)
        pos = np.clip(pos, 0, self.n - 1)
        ok = self.root_ids[pos] == ids
        if strict and not ok.all():
            raise KeyError(f"root ids not in v783: {ids[~ok].tolist()}")
        return pos[ok]

    # ----- annotation queries
    def neurons_of_type(self, name: str, side: str | None = None,
                        columns: Sequence[str] = ("cell_type", "hemibrain_type", "consolidated_type")) -> np.ndarray:
        """Indices of neurons whose type equals ``name`` in any of ``columns``.

        Hemibrain type cells like ``"LC40a,LC40b"`` are split on commas, so ``"LC40a"`` matches.
        """
        mask = np.zeros(self.n, dtype=bool)
        for c in columns:
            col = self.neurons[c].fillna("")
            mask |= col.str.split(",").apply(lambda xs: name in [x.strip() for x in xs]).to_numpy()
        if side is not None:
            mask &= (self.neurons["side"] == side).to_numpy()
        return np.flatnonzero(mask)

    def search_types(self, pattern: str) -> pd.DataFrame:
        """Regex search over all type columns; returns counts per (column, value)."""
        rx = re.compile(pattern, re.I)
        rows = []
        for c in ("cell_type", "hemibrain_type", "consolidated_type", "class", "sub_class"):
            vc = self.neurons[c].dropna().value_counts()
            for v, k in vc.items():
                if rx.search(str(v)):
                    rows.append((c, v, int(k)))
        return pd.DataFrame(rows, columns=["column", "value", "n"])

    def search_labels(self, pattern: str) -> pd.DataFrame:
        """Regex search over community labels. Returns (root_id, label) rows."""
        if self.labels is None:
            return pd.DataFrame(columns=["root_id", "label"])
        m = self.labels["label"].str.contains(pattern, case=False, regex=True, na=False)
        return self.labels.loc[m, ["root_id", "label"]]

    def neurons_with_label(self, pattern: str) -> np.ndarray:
        ids = self.search_labels(pattern)["root_id"].unique()
        return self.index_of(ids, strict=False)

    def super_class_mask(self, super_classes: Sequence[str]) -> np.ndarray:
        return self.neurons["super_class"].isin(list(super_classes)).to_numpy()

    def describe(self, idx: Iterable[int]) -> pd.DataFrame:
        idx = np.asarray(list(idx))
        cols = ["root_id", "super_class", "class", "sub_class", "cell_type", "hemibrain_type",
                "consolidated_type", "side", "nerve", "nt", "sign"]
        return self.neurons.iloc[idx][cols]

    # ----- stats
    def summary(self) -> str:
        W = self.W
        nsyn = int(W.data.sum())
        out_deg = np.diff(W.indptr)
        in_deg = np.bincount(W.indices, minlength=self.n)
        lines = [
            f"FlyWire v783 connectome",
            f"  neurons:            {self.n:,}",
            f"  connected pairs:    {W.nnz:,}",
            f"  synapses:           {nsyn:,}",
            f"  mean out-degree:    {out_deg.mean():.1f}  (max {out_deg.max():,})",
            f"  mean in-degree:     {in_deg.mean():.1f}  (max {in_deg.max():,})",
            f"  isolated neurons:   {int(((out_deg == 0) & (in_deg == 0)).sum()):,}",
            f"  neurotransmitter:   " + ", ".join(f"{k}={int(v):,}" for k, v in
                                                pd.Series(self.nt).value_counts().items()),
            f"  sign:               +1={int((self.sign > 0).sum()):,}  -1={int((self.sign < 0).sum()):,}",
            f"  super_class:        " + ", ".join(f"{k}={int(v):,}" for k, v in
                                                self.neurons["super_class"].value_counts(dropna=False).items()),
            f"  typed (cell_type):  {int(self.neurons['cell_type'].notna().sum()):,}",
            f"  typed (hemibrain):  {int(self.neurons['hemibrain_type'].notna().sum()):,}",
            f"  typed (consolid.):  {int(self.neurons['consolidated_type'].notna().sum()):,}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- build
def build(cfg: dict, verbose: bool = True) -> Connectome:
    """Build the connectome from the raw Codex tables (slow, ~1 min) without caching."""
    files = download(cfg, verbose=verbose)
    d = cfg["data"]
    log = (lambda *a: print(*a, file=sys.stderr)) if verbose else (lambda *a: None)

    log("reading neurons ...")
    neu = pd.read_csv(files["neurons"], usecols=["root_id", "group", "nt_type", "nt_type_score"])
    root_ids = np.sort(neu["root_id"].to_numpy(np.int64))
    n = len(root_ids)
    neu = neu.set_index("root_id").reindex(root_ids)

    log("reading connections ...")
    con = pd.read_csv(files["connections"], usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
                      dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32,
                             "nt_type": "category"})
    pre = np.searchsorted(root_ids, con["pre_root_id"].to_numpy())
    post = np.searchsorted(root_ids, con["post_root_id"].to_numpy())
    ok = (pre < n) & (post < n)
    ok &= (root_ids[np.clip(pre, 0, n - 1)] == con["pre_root_id"].to_numpy())
    ok &= (root_ids[np.clip(post, 0, n - 1)] == con["post_root_id"].to_numpy())
    if (~ok).any():
        log(f"  dropping {(~ok).sum():,} connection rows whose ids are not in neurons.csv")
    W = sp.coo_matrix((con["syn_count"].to_numpy()[ok], (pre[ok], post[ok])), shape=(n, n), dtype=np.int32)
    W = W.tocsr()          # duplicates (same pair in several neuropils) are summed
    W.sum_duplicates()
    W.sort_indices()

    # per-neuron neurotransmitter: neurons.csv prediction, else synapse-weighted majority of outputs
    nt = neu["nt_type"].astype("string").fillna("").to_numpy()
    missing = nt == ""
    if missing.any():
        sub = con.loc[ok & missing[pre], ["pre_root_id", "syn_count", "nt_type"]]
        maj = sub.groupby(["pre_root_id", "nt_type"], observed=True)["syn_count"].sum().unstack(fill_value=0)
        if len(maj):
            fb = maj.idxmax(axis=1)
            fb_idx = np.searchsorted(root_ids, fb.index.to_numpy())
            nt[fb_idx] = fb.to_numpy().astype(str)
            log(f"  filled {len(fb):,} missing nt predictions from output-synapse majority")
    nt[nt == ""] = "UNK"
    nt = nt.astype("<U4")
    sign_map = {k: int(v) for k, v in d["nt_sign"].items()}
    sign = np.array([sign_map.get(x, d["unknown_nt_sign"]) for x in nt], dtype=np.int8)

    log("reading annotations ...")
    cl = pd.read_csv(files["classification"]).set_index("root_id").reindex(root_ids)
    ct = pd.read_csv(files["consolidated_types"]).set_index("root_id").reindex(root_ids)
    names = pd.read_csv(files["names"]).set_index("root_id").reindex(root_ids)
    neurons = pd.DataFrame({
        "root_id": root_ids,
        "name": names["name"].to_numpy(),
        "group": names["group"].to_numpy(),
        "flow": cl["flow"].to_numpy(),
        "super_class": cl["super_class"].to_numpy(),
        "class": cl["class"].to_numpy(),
        "sub_class": cl["sub_class"].to_numpy(),
        "cell_type": cl["cell_type"].to_numpy(),
        "hemibrain_type": cl["hemibrain_type"].to_numpy(),
        "consolidated_type": ct["primary_type"].to_numpy(),
        "additional_types": ct["additional_type(s)"].to_numpy(),
        "hemilineage": cl["hemilineage"].to_numpy(),
        "side": cl["side"].to_numpy(),
        "nerve": cl["nerve"].to_numpy(),
        "nt": nt,
        "nt_score": neu["nt_type_score"].to_numpy(),
        "sign": sign,
    })
    for c in neurons.columns:
        if neurons[c].dtype == object:
            neurons[c] = neurons[c].astype("string")

    log("reading labels ...")
    labels = pd.read_csv(files["labels"], usecols=["root_id", "label", "user_name"])
    labels = labels[labels["root_id"].isin(root_ids)].reset_index(drop=True)
    labels["label"] = labels["label"].astype("string")

    return Connectome(root_ids=root_ids, W=W, nt=nt, sign=sign, neurons=neurons, labels=labels)


def cache_paths(cfg: dict) -> tuple[Path, Path, Path]:
    cd = resolve(cfg, cfg["data"]["cache_dir"])
    v = cfg["data"]["version"]
    return cd / f"connectome_v{v}.npz", cd / f"neurons_v{v}.parquet", cd / f"labels_v{v}.parquet"


def load_connectome(cfg: dict | None = None, rebuild: bool = False, verbose: bool = True) -> Connectome:
    """Load the cached connectome, building it from the raw tables if needed."""
    cfg = cfg or load_config()
    npz, pq, lq = cache_paths(cfg)
    if rebuild or not (npz.exists() and pq.exists() and lq.exists()):
        c = build(cfg, verbose=verbose)
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, root_ids=c.root_ids, indptr=c.W.indptr, indices=c.W.indices, data=c.W.data,
                 nt=c.nt, sign=c.sign)
        c.neurons.to_parquet(pq)
        c.labels.to_parquet(lq)
        return c
    z = np.load(npz)
    n = len(z["root_ids"])
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return Connectome(root_ids=z["root_ids"], W=W, nt=z["nt"], sign=z["sign"],
                      neurons=pd.read_parquet(pq), labels=pd.read_parquet(lq))


if __name__ == "__main__":  # python -m flyhamlet.data [--rebuild]
    c = load_connectome(rebuild="--rebuild" in sys.argv)
    print(c.summary())
