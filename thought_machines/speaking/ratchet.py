"""Speaking, part 2: language lets learning accumulate across generations.

Without a shared code, everything an animal learns dies with it, and each generation
starts again. With one, a generation can hand over what it found, and the next starts
from there. This is the cumulative-culture ratchet.

The task is deliberately abstract. There is a hidden target bit string. An individual
can test a bit flip and keep it if it helps, but has a limited lifetime of trials.
Fitness is the number of bits that match the target.

* `social = False`: each generation starts from a random string. Fitness plateaus at
  whatever one lifetime can reach.
* `social = True`: each generation receives the previous generation's final string as
  a message, received with some transmission errors, then continues improving it.
  Fitness climbs generation after generation, as long as one lifetime of learning
  gains more than the channel loses.
"""

from __future__ import annotations

import random


def lifetime_learning(start, target, trials: int, rng: random.Random):
    """Hill-climb by single bit flips for a limited number of trials."""
    current = list(start)
    score = sum(a == b for a, b in zip(current, target))
    for _ in range(trials):
        i = rng.randrange(len(current))
        current[i] ^= 1
        new = sum(a == b for a, b in zip(current, target))
        if new > score:
            score = new
        else:
            current[i] ^= 1  # undo
    return current, score


def transmit(message, error_rate: float, rng: random.Random):
    """A noisy channel: each bit of the message is flipped with probability `error_rate`."""
    return [b ^ 1 if rng.random() < error_rate else b for b in message]


def cumulative_culture(generations: int = 30, length: int = 100, trials: int = 60, error_rate: float = 0.0,
                       social: bool = True, seed: int = 0):
    """Returns the best fitness reached by each generation, as a fraction of the maximum."""
    rng = random.Random(seed)
    target = [rng.randrange(2) for _ in range(length)]
    message, history = None, []
    for _ in range(generations):
        if social and message is not None:
            start = transmit(message, error_rate, rng)
        else:
            start = [rng.randrange(2) for _ in range(length)]
        message, score = lifetime_learning(start, target, trials, rng)
        history.append(score / length)
    return history
