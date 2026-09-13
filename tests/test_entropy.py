"""Entropy estimators, whitening and typist tests against known distributions."""
import hashlib
import math

import numpy as np
import pytest
from scipy.stats import chisquare

from analysis.entropy import (K, ALPHABET, conditional_entropy_bits, encode, graph_structure, grid_random_walk,
                              lz_entropy_rate_bits, lz76_complexity, miller_madow_bits, plugin_entropy_bits,
                              transition_counts, log2_prob_under_model, clean_text)
from analysis.whiten import bytes_to_keys, hash_batches, whiten, REJECT_ABOVE
from analysis.typist import SuffixAutomaton, HamletTypist
from analysis.entropy_tests import most_common_value, collision_estimate, to_bits, _collision_expected_mean


def two_state_markov(n, p_stay, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.int64); x[0] = 0
    flips = rng.random(n) > p_stay
    for i in range(1, n):
        x[i] = x[i - 1] ^ int(flips[i])
    return x


# ------------------------------------------------------------------ marginal / conditional
def test_uniform_entropies():
    seq = np.random.default_rng(1).integers(0, K, 200_000)
    assert abs(plugin_entropy_bits(np.bincount(seq, minlength=K)) - math.log2(K)) < 0.01
    assert abs(miller_madow_bits(np.bincount(seq, minlength=K)) - math.log2(K)) < 0.01
    assert abs(conditional_entropy_bits(transition_counts(seq)) - math.log2(K)) < 0.02


def test_single_symbol_entropies_are_zero():
    seq = np.full(5000, 3)
    assert plugin_entropy_bits(np.bincount(seq, minlength=K)) == 0.0
    assert miller_madow_bits(np.bincount(seq, minlength=K)) == 0.0
    assert conditional_entropy_bits(transition_counts(seq)) == 0.0
    assert lz_entropy_rate_bits(seq) < 0.05


def test_miller_madow_corrects_upward_and_matches_formula():
    counts = np.array([5, 3, 2, 0, 0])
    n, m = counts.sum(), 3
    assert math.isclose(miller_madow_bits(counts), plugin_entropy_bits(counts) + (m - 1) / (2 * n) / math.log(2))
    assert miller_madow_bits(counts) > plugin_entropy_bits(counts)


def test_two_state_markov_conditional_entropy():
    p = 0.9
    x = two_state_markov(300_000, p, seed=2)
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))       # 0.469 bits
    T = transition_counts(x, k=2)
    assert abs(conditional_entropy_bits(T) - h) < 0.01
    assert abs(plugin_entropy_bits(np.bincount(x)) - 1.0) < 0.01      # marginal is uniform


# ------------------------------------------------------------------ Lempel-Ziv
def test_lz76_complexity_known_values():
    assert lz76_complexity("0001101001000101") == 6      # Kaspar & Schuster 1987 example
    assert lz76_complexity("1001111011000010") == 6      # Lempel & Ziv 1976, Fig. 1
    assert lz76_complexity("aaaaaaaa") == 2
    assert lz76_complexity("a") == 1


def test_lz_rate_orders_sources_correctly():
    n = 100_000
    uni = np.random.default_rng(3).integers(0, K, n)
    const = np.zeros(n, dtype=np.int64)
    mk = two_state_markov(n, 0.9, seed=4)
    h_uni, h_mk, h_const = lz_entropy_rate_bits(uni), lz_entropy_rate_bits(mk), lz_entropy_rate_bits(const)
    assert h_const < 0.01 < h_mk < h_uni
    assert 0.4 < h_mk < 1.0            # true rate 0.469; LZ76 is biased upward at this length
    assert 4.0 < h_uni < 5.5           # true rate 4.755


# ------------------------------------------------------------------ graph structure & null walk
def test_graph_structure():
    seq = encode("abababab")
    T = transition_counts(seq); vis = np.bincount(seq, minlength=K)
    g = graph_structure(T, vis)
    assert g["n_visited"] == 2 and g["single_scc"] and g["absorbing_keys"] == [] and len(g["zero_visit_keys"]) == K - 2
    seq = encode("abccc")
    T = transition_counts(seq); vis = np.bincount(seq, minlength=K)
    g = graph_structure(T, vis)
    assert g["absorbing_keys"] == ["c"] and not g["single_scc"]


def test_grid_walk_is_uniform_on_neighbours():
    w = grid_random_walk(3, 9, 200_000, seed=5)
    T = transition_counts(w, k=27)
    for i in range(27):
        r, c = divmod(i, 9)
        nbrs = [j for j in range(27) if (abs(divmod(j, 9)[0] - r) + abs(divmod(j, 9)[1] - c)) == 1]
        row = T[i]
        assert set(np.flatnonzero(row)) == set(nbrs)
        assert row[nbrs].min() > 0.7 * row[nbrs].mean()


# ------------------------------------------------------------------ Hamlet scoring
def test_log2_prob_and_zero_pairs():
    text = "abab"
    p0 = np.zeros(K); p0[IDX_A := 0] = 0.5; p0[1] = 0.5
    P = np.zeros((K, K)); P[0, 1] = 1.0; P[1, 0] = 1.0
    lp, zero, ntr = log2_prob_under_model(text, p0, P)
    assert zero == [] and ntr == 3 and math.isclose(lp, math.log2(0.5))
    lp, zero, ntr = log2_prob_under_model("abc", p0, P)
    assert lp == -math.inf and zero == [("bc", 1)]


def test_clean_text():
    assert clean_text("To be,\nor NOT   to be!") == "to be or not to be"


# ------------------------------------------------------------------ whitening
def test_hash_batches_is_sha256_of_64_uint32():
    isis = np.arange(130, dtype=np.uint32)
    h = hash_batches(isis)
    assert len(h) == 64                                   # 2 full batches, remainder dropped
    assert h[:32] == hashlib.sha256(isis[:64].astype("<u4").tobytes()).digest()


def test_rejection_sampling_mapping_and_uniformity():
    assert REJECT_ABOVE == 9 * 27
    b = bytes(range(256))
    keys = bytes_to_keys(b)
    assert len(keys) == 243 and keys.tolist() == [i % 27 for i in range(243)]
    # each key is hit exactly 9 times by the accepted byte values: exact uniformity of the map
    assert np.bincount(keys, minlength=27).tolist() == [9] * 27
    # statistical uniformity on hashed data
    rng = np.random.default_rng(6)
    isis = rng.integers(1, 5000, 64 * 20000).astype(np.uint32)
    text, h, st = whiten(isis)
    counts = np.bincount(np.fromiter((ALPHABET.index(c) for c in text), dtype=int), minlength=27)
    assert chisquare(counts).pvalue > 0.001
    assert abs(st["accept_fraction"] - 243 / 256) < 0.01


# ------------------------------------------------------------------ typist
def test_suffix_automaton_longest_match():
    hamlet = "to be or not to be that is the question"
    sam = SuffixAutomaton(hamlet)
    assert sam.contains("not to be") and not sam.contains("to bee")
    t = HamletTypist(hamlet)
    t.feed("xxthat is thexx", verbose=False)
    assert t.record == len("that is the") and t.record_text == "that is the"
    assert t.record_pos_hamlet == hamlet.index("that is the")


# ------------------------------------------------------------------ NIST 800-90B
def test_mcv_known_distributions():
    uni = np.random.default_rng(7).integers(0, 2, 100_000)
    assert 0.95 < most_common_value(uni)["min_entropy_bits"] <= 1.0
    assert most_common_value(np.zeros(1000, dtype=int))["min_entropy_bits"] == 0.0
    biased = (np.random.default_rng(8).random(200_000) < 0.75).astype(int)
    assert abs(most_common_value(biased)["min_entropy_bits"] - (-math.log2(0.75))) < 0.03


def test_collision_formula_and_estimate():
    assert math.isclose(_collision_expected_mean(0.5), 2.5)
    assert abs(_collision_expected_mean(0.9) - (2 * (0.81 + 0.01) + 3 * 0.18)) < 1e-9
    fair = np.random.default_rng(9).integers(0, 2, 300_000)
    assert collision_estimate(fair)["min_entropy_bits"] > 0.85   # conservative near p=0.5 (flat optimum)
    biased = (np.random.default_rng(10).random(300_000) < 0.8).astype(int)
    r = collision_estimate(biased)
    assert abs(r["p"] - 0.8) < 0.02 and abs(r["min_entropy_bits"] - (-math.log2(0.8))) < 0.05
    assert collision_estimate(np.zeros(1000, dtype=int))["min_entropy_bits"] < 1e-4


def test_to_bits():
    bits, w = to_bits(np.array([5, 1]))
    assert w == 3 and bits.tolist() == [1, 0, 1, 0, 0, 1]
