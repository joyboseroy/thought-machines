"""RuleLogit: a small hybrid in the direction Domingos sketches.

The structure comes from a symbolic learner: a decision tree carves the input
space into rules. The parameters are probabilistic: each rule gets a weight in a
log-linear model, fitted by gradient descent on log loss with an L2 penalty.

Compared with the raw tree this changes one thing that matters under noise. A
rule backed by a single example gets a small weight and defers to the global
bias, while a rule backed by many consistent examples gets a large one.

This is a toy, not a Markov logic network. There is no first-order logic and no
joint inference. It shows the shape of the idea: symbolic representation,
probabilistic evaluation, gradient-based optimisation.
"""

from __future__ import annotations

import math

from ..symbolist.decision_tree import DecisionTree


def _sigmoid(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class RuleLogit:
    def __init__(self, l2: float = 0.02, lr: float = 1.0, epochs: int = 400, max_depth: int | None = None):
        self.l2, self.lr, self.epochs, self.max_depth = l2, lr, epochs, max_depth

    def fit(self, X, y):
        self.tree = DecisionTree(max_depth=self.max_depth).fit(X, y)
        ids = [self.tree.apply(x) for x in X]
        n = len(y)
        self.w = {i: 0.0 for i in set(ids)}
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w = {i: self.l2 * self.w[i] for i in self.w}
            grad_b = 0.0
            for i, t in zip(ids, y):
                err = _sigmoid(self.b + self.w[i]) - t
                grad_w[i] += err / n
                grad_b += err / n
            for i in self.w:
                self.w[i] -= self.lr * grad_w[i]
            self.b -= self.lr * grad_b
        return self

    def proba(self, x) -> float:
        return _sigmoid(self.b + self.w.get(self.tree.apply(x), 0.0))

    def predict(self, x) -> int:
        return 1 if self.proba(x) > 0.5 else 0

    def predict_all(self, X):
        return [self.predict(x) for x in X]

    def n_rules(self) -> int:
        return len(self.w)
