"""Metropolis sampling: inference when the posterior cannot be written in closed form.

The sampler only needs the posterior up to a constant, so it never has to
compute the integral that makes exact Bayesian inference hard.
"""

from __future__ import annotations

import math
import random


def metropolis(log_p, x0: float, steps: int = 20000, step: float = 0.1,
               burn: int = 2000, seed: int = 0):
    """Random-walk Metropolis. Returns (samples after burn-in, acceptance rate).

    log_p may return -inf outside the support.
    """
    rng = random.Random(seed)
    x, lp = x0, log_p(x0)
    samples, accepted = [], 0
    for i in range(steps):
        y = x + rng.gauss(0, step)
        lq = log_p(y)
        if lq >= lp or rng.random() < math.exp(lq - lp):
            x, lp = y, lq
            accepted += 1
        if i >= burn:
            samples.append(x)
    return samples, accepted / steps


def coin_log_posterior(heads: int, tails: int, a: float = 1.0, b: float = 1.0):
    """Unnormalised log posterior of a coin's bias under a Beta(a, b) prior."""
    def log_p(theta):
        if not 0 < theta < 1:
            return float("-inf")
        return (a - 1 + heads) * math.log(theta) + (b - 1 + tails) * math.log(1 - theta)
    return log_p
