"""Active learning: choose the experiment, then update, then choose again.

Given a belief over hypotheses and a set of possible experiments, pick the one
with the highest expected information gain (expected drop in entropy). With a
hidden threshold and yes/no questions this rediscovers binary search without
being told about it.
"""

from __future__ import annotations

import math
import random


def entropy(dist: dict) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def update(prior: dict, p1, query, outcome: int) -> dict:
    """Bayes update. p1(h, query) = P(outcome = 1 | h, query). Zero-mass hypotheses are dropped."""
    post = {}
    for h, p in prior.items():
        like = p1(h, query) if outcome == 1 else 1 - p1(h, query)
        if p * like > 0:
            post[h] = p * like
    z = sum(post.values())
    return {h: p / z for h, p in post.items()}


def expected_info_gain(prior: dict, p1, query) -> float:
    p_yes = sum(p * p1(h, query) for h, p in prior.items())
    gain = entropy(prior)
    for outcome, w in ((1, p_yes), (0, 1 - p_yes)):
        if w > 0:
            gain -= w * entropy(update(prior, p1, query, outcome))
    return gain


def best_query(prior: dict, p1, queries):
    return max(queries, key=lambda q: expected_info_gain(prior, p1, q))


def run_session(prior: dict, p1, queries, answer, chooser: str = "eig", max_queries: int = 40,
                confidence: float = 0.99, seed: int = 0):
    """Ask questions until one hypothesis holds `confidence` of the mass.

    `answer(query)` returns the true outcome. Returns (asked queries, final belief)."""
    rng = random.Random(seed)
    asked, belief = [], dict(prior)
    while len(asked) < max_queries and max(belief.values()) < confidence:
        q = best_query(belief, p1, queries) if chooser == "eig" else rng.choice(queries)
        asked.append(q)
        belief = update(belief, p1, q, answer(q))
    return asked, belief


def threshold_world(n: int, noise: float = 0.0):
    """Hypotheses t in 0..n-1. Query q asks 'is t <= q?'. Optional answer noise."""
    prior = {t: 1.0 / n for t in range(n)}

    def p1(t, q):
        return (1 - noise) if t <= q else noise

    return prior, p1, list(range(n))
