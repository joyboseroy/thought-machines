"""Naive Bayes: Bayes' rule with the assumption that features are independent given the class."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.n = len(y)
        self.class_counts = Counter(y)
        self.values = defaultdict(set)
        self.counts = defaultdict(Counter)  # (class, feature) -> value counts
        for x, label in zip(X, y):
            for f, v in enumerate(x):
                self.values[f].add(v)
                self.counts[(label, f)][v] += 1
        return self

    def predict_proba(self, x) -> dict:
        logp = {}
        for c, nc in self.class_counts.items():
            lp = math.log(nc / self.n)
            for f, v in enumerate(x):
                k = len(self.values[f]) + (0 if v in self.values[f] else 1)
                lp += math.log((self.counts[(c, f)][v] + self.alpha) / (nc + self.alpha * k))
            logp[c] = lp
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {c: math.exp(v - m) / z for c, v in logp.items()}

    def predict(self, x):
        p = self.predict_proba(x)
        return max(sorted(p), key=p.get)  # ties break toward the smaller label, deterministically

    def predict_all(self, X):
        return [self.predict(x) for x in X]
