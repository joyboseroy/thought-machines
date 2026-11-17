"""Concepts as points in a space. Toy hand-built features, not learned embeddings.

The vectors are invented for illustration. The point is the mechanism: once
concepts are vectors, similarity becomes geometry and categories become regions.
"""

from __future__ import annotations

import math

FEATURES = ("animate", "furry", "domestic", "large", "wheels", "engine", "carnivore")

VECTORS = {
    "dog": (1, 1, 1, 0.3, 0, 0, 0.6),
    "cat": (1, 1, 1, 0.1, 0, 0, 0.9),
    "wolf": (1, 1, 0, 0.4, 0, 0, 1.0),
    "lion": (1, 1, 0, 0.8, 0, 0, 1.0),
    "car": (0, 0, 0, 0.5, 1, 1, 0),
    "truck": (0, 0, 0, 0.9, 1, 1, 0),
}

ANIMALS = ("dog", "cat", "wolf", "lion")
VEHICLES = ("car", "truck")


def cosine(u, v) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu, nv = math.sqrt(sum(a * a for a in u)), math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv)


def similarity_matrix(vectors=VECTORS) -> dict:
    return {a: {b: cosine(vectors[a], vectors[b]) for b in vectors} for a in vectors}


def nearest(name: str, k: int = 3, vectors=VECTORS):
    sims = [(other, cosine(vectors[name], v)) for other, v in vectors.items() if other != name]
    return sorted(sims, key=lambda t: -t[1])[:k]
