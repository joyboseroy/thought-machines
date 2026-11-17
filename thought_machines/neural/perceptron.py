"""A single threshold neuron, trained with the perceptron rule."""

from __future__ import annotations

import random

GATES = {
    "AND": [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)],
    "OR": [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)],
    "XOR": [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)],
}


class Perceptron:
    def __init__(self, n_inputs: int, seed: int = 0):
        rng = random.Random(seed)
        self.w = [rng.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.b = 0.0

    def predict(self, x) -> int:
        return 1 if sum(w * xi for w, xi in zip(self.w, x)) + self.b > 0 else 0

    def accuracy(self, data) -> float:
        return sum(self.predict(x) == y for x, y in data) / len(data)

    def fit(self, data, epochs: int = 200, lr: float = 0.1):
        """Returns the number of errors in each epoch. Stops early if an epoch has none."""
        history = []
        for _ in range(epochs):
            errors = 0
            for x, y in data:
                err = y - self.predict(x)
                if err:
                    errors += 1
                    self.w = [w + lr * err * xi for w, xi in zip(self.w, x)]
                    self.b += lr * err
            history.append(errors)
            if errors == 0:
                break
        return history
