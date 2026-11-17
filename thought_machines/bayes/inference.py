"""Bayes' rule and belief updating, from scratch."""

from __future__ import annotations


def posterior(prior: float, sensitivity: float, false_positive_rate: float) -> float:
    """P(disease | positive test)."""
    hit = prior * sensitivity
    miss = (1 - prior) * false_positive_rate
    return hit / (hit + miss)


def posterior_negative(prior: float, sensitivity: float, false_positive_rate: float) -> float:
    """P(disease | negative test)."""
    hit = prior * (1 - sensitivity)
    miss = (1 - prior) * (1 - false_positive_rate)
    return hit / (hit + miss)


def sequential_update(prior: float, sensitivity: float, false_positive_rate: float, results):
    """Update belief after each test result (True = positive). Returns the whole trajectory."""
    belief, path = prior, [prior]
    for positive in results:
        f = posterior if positive else posterior_negative
        belief = f(belief, sensitivity, false_positive_rate)
        path.append(belief)
    return path


def normalise(weights: dict) -> dict:
    total = sum(weights.values())
    if total == 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def update(prior: dict, likelihood) -> dict:
    """General discrete update: prior over hypotheses, likelihood(h) = P(evidence | h)."""
    return normalise({h: p * likelihood(h) for h, p in prior.items()})
