"""Predictive coding: perception as minimising prediction error.

The brain (on this view) holds a generative model: a hidden cause v produces an
observation u through u = g(v) + noise, with a prior on v. Perception is then
inference on v. Here inference is done by gradient descent on the two prediction
errors, which is a neurally plausible recipe, and is checked against the exact
posterior computed on a grid.
"""

from __future__ import annotations

import math


def g(v: float) -> float:
    return v * v


def dg(v: float) -> float:
    return 2 * v


def log_posterior(v, u, prior_mean, prior_var, noise_var):
    return -((v - prior_mean) ** 2) / (2 * prior_var) - ((u - g(v)) ** 2) / (2 * noise_var)


def infer_by_gradient(u: float, prior_mean: float, prior_var: float, noise_var: float,
                      steps: int = 4000, dt: float = 0.01, v0: float | None = None):
    """Descend on prediction errors. Returns the trajectory of v."""
    v = prior_mean if v0 is None else v0
    path = [v]
    for _ in range(steps):
        err_obs = (u - g(v)) / noise_var   # how wrong the sensory prediction is
        err_prior = (v - prior_mean) / prior_var  # how far the belief is from the prior
        v += dt * (err_obs * dg(v) - err_prior)
        path.append(v)
    return path


def grid_map(u, prior_mean, prior_var, noise_var, lo=0.0, hi=6.0, n=60001):
    """Brute-force posterior mode on a grid, for checking the gradient answer."""
    best, best_v = -math.inf, lo
    for i in range(n):
        v = lo + (hi - lo) * i / (n - 1)
        lp = log_posterior(v, u, prior_mean, prior_var, noise_var)
        if lp > best:
            best, best_v = lp, v
    return best_v


def thermostat(target: float = 22.0, start: float = 10.0, leak: float = 0.5, gain: float = 2.0,
               steps: int = 60, dt: float = 0.1):
    """Toy active inference: the agent expects `target`, and acts to make the world match.

    Perception is trivial here (it reads the temperature directly). Action is chosen to
    reduce the prediction error, against a constant heat leak. Returns the temperature path."""
    temp, path = start, [start]
    for _ in range(steps):
        error = temp - target
        action = -gain * error
        temp += dt * (action - leak)
        path.append(temp)
    return path
