"""Kernel methods: similarity as the only thing the learner ever looks at.

`KernelSVM` minimises the SVM objective (L2 penalty plus hinge loss) in the dual,
using the kernelised Pegasos update. Labels are 0/1 on the outside. There is no
bias term, which is fine for the symmetric toy problems used here.
"""

from __future__ import annotations

import math
import random


def linear(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def rbf(gamma: float = 1.0):
    return lambda a, b: math.exp(-gamma * sum((x - y) ** 2 for x, y in zip(a, b)))


def poly(degree: int = 2, c: float = 1.0):
    return lambda a, b: (linear(a, b) + c) ** degree


class KernelSVM:
    def __init__(self, kernel=linear, lam: float = 0.01, epochs: int = 30, seed: int = 0):
        self.kernel, self.lam, self.epochs, self.seed = kernel, lam, epochs, seed

    def fit(self, X, y):
        X = [tuple(x) for x in X]
        t = [1 if v == 1 else -1 for v in y]
        n = len(X)
        gram = [[self.kernel(a, b) for b in X] for a in X]
        alpha = [0] * n
        rng = random.Random(self.seed)
        step = 0
        for _ in range(self.epochs):
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                step += 1
                s = sum(alpha[j] * t[j] * gram[j][i] for j in range(n) if alpha[j]) / (self.lam * step)
                if t[i] * s < 1:
                    alpha[i] += 1
        scale = 1.0 / (self.lam * step)
        self.support = [(X[j], alpha[j] * t[j] * scale) for j in range(n) if alpha[j]]
        return self

    def decision(self, x) -> float:
        return sum(c * self.kernel(sv, x) for sv, c in self.support)

    def predict(self, x) -> int:
        return 1 if self.decision(x) > 0 else 0

    def predict_all(self, X):
        return [self.predict(x) for x in X]

    def n_support(self) -> int:
        return len(self.support)


def make_circles(n: int = 100, noise: float = 0.05, seed: int = 0):
    """Inner ring is class 0, outer ring is class 1."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        label = i % 2
        r = (0.5 if label == 0 else 1.0) + rng.gauss(0, noise)
        a = rng.uniform(0, 2 * math.pi)
        X.append((r * math.cos(a), r * math.sin(a)))
        y.append(label)
    return X, y


def accuracy(model, X, y) -> float:
    return sum(p == t for p, t in zip(model.predict_all(X), y)) / len(y)
