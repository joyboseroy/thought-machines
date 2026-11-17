"""k-nearest neighbours: classify by similarity to remembered examples."""

from __future__ import annotations

import math


def euclidean(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def hamming(a, b) -> float:
    return sum(x != y for x, y in zip(a, b))


class KNN:
    def __init__(self, k: int = 3, distance=euclidean):
        self.k, self.distance = k, distance
        self.X, self.y = [], []

    def fit(self, X, y):
        self.X, self.y = [tuple(x) for x in X], list(y)
        return self

    def predict(self, x):
        near = sorted(((self.distance(x, xi), i) for i, xi in enumerate(self.X)))[: self.k]
        votes: dict = {}
        for _, i in near:
            votes[self.y[i]] = votes.get(self.y[i], 0) + 1
        top = max(votes.values())
        for _, i in near:  # ties go to the label of the closest neighbour
            if votes[self.y[i]] == top:
                return self.y[i]

    def predict_all(self, X):
        return [self.predict(x) for x in X]
