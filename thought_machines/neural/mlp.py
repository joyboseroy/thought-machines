"""A one-hidden-layer network with hand-written backpropagation. No numpy."""

from __future__ import annotations

import math
import random


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class MLP:
    def __init__(self, n_in: int, n_hidden: int, seed: int = 0):
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.w2 = [rng.uniform(-1, 1) for _ in range(n_hidden)]
        self.b2 = 0.0

    def forward(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.w1, self.b1)]
        return h, sigmoid(sum(w * hi for w, hi in zip(self.w2, h)) + self.b2)

    def predict_proba(self, x) -> float:
        return self.forward(x)[1]

    def predict(self, x) -> int:
        return 1 if self.predict_proba(x) > 0.5 else 0

    def accuracy(self, data) -> float:
        return sum(self.predict(x) == y for x, y in data) / len(data)

    def loss(self, data) -> float:
        eps = 1e-12
        total = 0.0
        for x, y in data:
            p = self.predict_proba(x)
            total -= y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps)
        return total / len(data)

    def fit(self, data, epochs: int = 3000, lr: float = 0.5, seed: int = 0, log_every: int = 500):
        """Stochastic gradient descent on cross-entropy. Returns sampled (epoch, loss) pairs."""
        rng = random.Random(seed)
        data = list(data)
        history = []
        for epoch in range(epochs):
            rng.shuffle(data)
            for x, y in data:
                h, p = self.forward(x)
                dz = p - y  # gradient of cross-entropy w.r.t. the output pre-activation
                dh = [dz * w * (1 - hi * hi) for w, hi in zip(self.w2, h)]
                for j, hi in enumerate(h):
                    self.w2[j] -= lr * dz * hi
                    for i, xi in enumerate(x):
                        self.w1[j][i] -= lr * dh[j] * xi
                    self.b1[j] -= lr * dh[j]
                self.b2 -= lr * dz
            if epoch % log_every == 0 or epoch == epochs - 1:
                history.append((epoch, self.loss(data)))
        return history
