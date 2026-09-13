#!/usr/bin/env python3
"""Consume a key stream and track the longest substring of Hamlet it contains.

Uses a suffix automaton built over the cleaned text of Hamlet: streaming each key
through it gives, in amortised O(1), the length of the longest suffix of the
output so far that occurs anywhere in Hamlet.

    python analysis/typist.py keys.txt            # or:  ... | python analysis/typist.py -
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class SuffixAutomaton:
    """Minimal suffix automaton (Blumer et al. 1985) with a streaming longest-match matcher."""

    def __init__(self, text: str = ""):
        self.next: list[dict] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.first_end: list[int] = [0]        # end position in text of the first occurrence
        self.last = 0
        self.n = 0
        self.state = 0
        self.match_len = 0
        for ch in text:
            self.extend(ch)

    def extend(self, ch: str):
        """Append one symbol to the indexed text (online construction)."""
        nxt, link, length, fe = self.next, self.link, self.length, self.first_end
        cur = len(length)
        length.append(length[self.last] + 1); link.append(0); nxt.append({}); fe.append(self.n + 1)
        self.n += 1
        p = self.last
        while p != -1 and ch not in nxt[p]:
            nxt[p][ch] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][ch]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1); nxt.append(dict(nxt[q])); link.append(link[q]); fe.append(fe[q])
                while p != -1 and nxt[p].get(ch) == q:
                    nxt[p][ch] = clone
                    p = link[p]
                link[q] = clone; link[cur] = clone
        self.last = cur

    def feed(self, ch: str) -> int:
        """Advance by one symbol; return the longest suffix length that is a substring of the text."""
        while self.state != -1 and ch not in self.next[self.state]:
            self.state = self.link[self.state]
            if self.state != -1:
                self.match_len = self.length[self.state]
        if self.state == -1:
            self.state, self.match_len = 0, 0
        else:
            self.state = self.next[self.state][ch]
            self.match_len += 1
        return self.match_len

    def contains(self, s: str) -> bool:
        st = 0
        for ch in s:
            st = self.next[st].get(ch)
            if st is None:
                return False
        return True


class HamletTypist:
    def __init__(self, hamlet: str):
        self.hamlet = hamlet
        self.sam = SuffixAutomaton(hamlet)
        self.record = 0; self.record_text = ""; self.record_pos_stream = -1; self.record_pos_hamlet = -1
        self.n = 0; self.t0 = time.perf_counter()

    def feed(self, keys: str, verbose: bool = True):
        for ch in keys:
            self.n += 1
            m = self.sam.feed(ch)
            if m > self.record:
                self.record = m
                self.record_pos_stream = self.n
                self.record_pos_hamlet = self.sam.first_end[self.sam.state] - m
                self.record_text = self.hamlet[self.record_pos_hamlet:self.record_pos_hamlet + m]
                if verbose:
                    print(f"[key {self.n:>9,}] new record: {m:3d} chars {self.record_text!r} (Hamlet offset {self.record_pos_hamlet:,})", file=sys.stderr)
        return self.record

    def keys_per_s(self) -> float:
        return self.n / max(1e-9, time.perf_counter() - self.t0)

    def summary(self) -> str:
        return (f"keys consumed: {self.n:,}; longest Hamlet substring: {self.record} chars {self.record_text!r} "
                f"(at key {self.record_pos_stream:,}, Hamlet offset {self.record_pos_hamlet:,}); {self.keys_per_s():,.0f} keys/s")


def main(argv=None):
    from flyhamlet.config import load_config, resolve
    from analysis.entropy import fetch_hamlet
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", help="key text file, or '-' for stdin")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    hamlet = fetch_hamlet(cfg["analysis"]["hamlet_url"], resolve(cfg, cfg["analysis"]["hamlet_cache"]))
    t = HamletTypist(hamlet)
    src = sys.stdin if a.keys == "-" else open(a.keys)
    while True:
        chunk = src.read(1 << 16)
        if not chunk:
            break
        t.feed(chunk)
    print(t.summary())


if __name__ == "__main__":
    main()
