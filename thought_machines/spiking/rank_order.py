"""Rank-order coding: information in the order in which neurons fire.

Stronger inputs fire earlier. A downstream neuron reads only the firing order,
weighting each arriving spike by a factor that shrinks with its rank. Because it
uses order and not magnitudes, the code ignores any monotonic distortion of the
inputs, such as a change of overall gain or contrast.
"""

from __future__ import annotations

import math
import random


def firing_order(intensities):
    """Indices from strongest to weakest, ties broken by index."""
    return sorted(range(len(intensities)), key=lambda i: (-intensities[i], i))


def ranks(order):
    r = [0] * len(order)
    for pos, idx in enumerate(order):
        r[idx] = pos
    return r


def similarity(order_a, order_b, mod: float = 0.9) -> float:
    """Thorpe-style match in [0, 1]. 1 means identical firing order."""
    ra = ranks(order_a)
    num = sum(mod ** pos * mod ** ra[idx] for pos, idx in enumerate(order_b))
    den = sum(mod ** (2 * pos) for pos in range(len(order_a)))
    return num / den


class RankOrderClassifier:
    def __init__(self, mod: float = 0.9):
        self.mod = mod
        self.templates: dict = {}

    def fit(self, templates: dict):
        self.templates = {k: firing_order(v) for k, v in templates.items()}
        return self

    def predict(self, intensities):
        order = firing_order(intensities)
        return max(self.templates, key=lambda k: similarity(self.templates[k], order, self.mod))


def nearest_euclid(templates: dict, x):
    return min(templates, key=lambda k: sum((a - b) ** 2 for a, b in zip(templates[k], x)))


def nearest_cosine(templates: dict, x):
    def cos(a, b):
        return sum(p * q for p, q in zip(a, b)) / (math.sqrt(sum(p * p for p in a)) * math.sqrt(sum(q * q for q in b)))
    return max(templates, key=lambda k: cos(templates[k], x))


def make_task(n_classes: int = 8, dim: int = 30, spread: float = 0.12, seed: int = 0):
    """Similar templates: a shared base pattern plus a small class-specific perturbation."""
    rng = random.Random(seed)
    base = [rng.random() for _ in range(dim)]
    return {c: [min(1.0, max(0.01, b + rng.gauss(0, spread))) for b in base] for c in range(n_classes)}


def distortion_experiment(trials: int = 400, seed: int = 0, noise: float = 0.01) -> dict:
    """Classify stimuli after a random monotonic distortion x -> x**gamma, gamma in [0.3, 3].

    Returns accuracy for rank-order, Euclidean and cosine template matching."""
    rng = random.Random(seed)
    templates = make_task(seed=seed)
    clf = RankOrderClassifier().fit(templates)
    hits = {"rank order": 0, "euclidean": 0, "cosine": 0}
    for _ in range(trials):
        label = rng.randrange(len(templates))
        gamma = rng.uniform(0.3, 3.0)
        x = [v ** gamma + rng.gauss(0, noise) for v in templates[label]]
        hits["rank order"] += clf.predict(x) == label
        hits["euclidean"] += nearest_euclid(templates, x) == label
        hits["cosine"] += nearest_cosine(templates, x) == label
    return {k: v / trials for k, v in hits.items()}
