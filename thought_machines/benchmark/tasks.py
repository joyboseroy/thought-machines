"""Hidden-rule tasks over 8 binary features.

Each task labels an input by a rule that depends on only a few features. The
other features are distractors. There are exactly 256 possible inputs, so a
learner trained on n examples is tested on all 256 - n it has not seen, and
generalisation is measured exactly rather than estimated.
"""

from __future__ import annotations

import random
from itertools import product

N_FEATURES = 8
ALL_INPUTS = list(product((0, 1), repeat=N_FEATURES))

TASKS = {
    "and": lambda x: x[0] & x[1],
    "dnf": lambda x: (x[0] & x[1]) | x[2],
    "majority": lambda x: int(x[0] + x[1] + x[2] >= 2),
    "xor": lambda x: x[0] ^ x[1],
    "parity3": lambda x: x[0] ^ x[1] ^ x[2],
}

DESCRIPTIONS = {
    "and": "x0 AND x1",
    "dnf": "(x0 AND x1) OR x2",
    "majority": "at least two of x0, x1, x2",
    "xor": "x0 XOR x1",
    "parity3": "x0 XOR x1 XOR x2",
}


def make_split(task: str, n_train: int, noise: float, rng: random.Random):
    """Returns (X_train, y_train_noisy, X_test, y_test_clean)."""
    rule = TASKS[task]
    train = rng.sample(ALL_INPUTS, n_train)
    held = set(train)
    test = [x for x in ALL_INPUTS if x not in held]
    y_train = [rule(x) ^ (1 if rng.random() < noise else 0) for x in train]
    return train, y_train, test, [rule(x) for x in test]
