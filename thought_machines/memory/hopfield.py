"""Hopfield network: content-addressable memory as energy minimisation.

Patterns are vectors of +1/-1. Storing uses the Hebbian outer-product rule.
Recall starts from a damaged cue and flips units one at a time, each flip
lowering the network's energy, until the state settles.
"""

from __future__ import annotations

import random


class Hopfield:
    def __init__(self, n: int):
        self.n = n
        self.w = [[0.0] * n for _ in range(n)]

    def store(self, patterns):
        for p in patterns:
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.w[i][j] += p[i] * p[j] / self.n
        return self

    def energy(self, s) -> float:
        return -0.5 * sum(s[i] * self.w[i][j] * s[j] for i in range(self.n) for j in range(self.n))

    def recall(self, cue, max_sweeps: int = 20, seed: int = 0):
        rng = random.Random(seed)
        s = list(cue)
        order = list(range(self.n))
        for _ in range(max_sweeps):
            rng.shuffle(order)
            changed = False
            for i in order:
                h = sum(self.w[i][j] * s[j] for j in range(self.n))
                new = 1 if h > 0 else (-1 if h < 0 else s[i])
                if new != s[i]:
                    s[i], changed = new, True
            if not changed:
                break
        return s


def random_pattern(n: int, rng: random.Random):
    return [rng.choice((-1, 1)) for _ in range(n)]


def corrupt(pattern, fraction: float, rng: random.Random):
    idx = set(rng.sample(range(len(pattern)), int(round(fraction * len(pattern)))))
    return [-v if i in idx else v for i, v in enumerate(pattern)]


def recall_rate(n: int, n_patterns: int, noise: float = 0.1, trials: int = 3, seed: int = 0) -> float:
    """Fraction of stored patterns recovered exactly from a noisy cue."""
    rng = random.Random(seed)
    ok = total = 0
    for _ in range(trials):
        pats = [random_pattern(n, rng) for _ in range(n_patterns)]
        net = Hopfield(n).store(pats)
        for p in pats:
            ok += net.recall(corrupt(p, noise, rng), seed=rng.randint(0, 10**6)) == p
            total += 1
    return ok / total
