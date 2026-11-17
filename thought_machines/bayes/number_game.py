"""The number game: inductive inference with an explicit hypothesis space.

Someone says "16" is an example of a concept in 1..100. What else is? Many
hypotheses fit (even numbers, squares, powers of two, intervals around 16, ...).
Data alone cannot choose, so the answer depends on two things:

  * the prior over hypotheses (inductive bias), and
  * the likelihood. Under "strong sampling", examples are drawn from the concept,
    so a small concept that fits the data beats a large one (the size principle).

Same observations plus different priors gives different conclusions, which is the
whole point of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

DOMAIN = tuple(range(1, 101))


def _is_prime(n):
    return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))


MATHY = {
    "even": lambda n: n % 2 == 0,
    "odd": lambda n: n % 2 == 1,
    "squares": lambda n: int(n ** 0.5) ** 2 == n,
    "cubes": lambda n: round(n ** (1 / 3)) ** 3 == n,
    "powers of 2": lambda n: n & (n - 1) == 0,
    "powers of 10": lambda n: n in (1, 10, 100),
    "primes": _is_prime,
    "multiples of 3": lambda n: n % 3 == 0,
    "multiples of 5": lambda n: n % 5 == 0,
    "multiples of 7": lambda n: n % 7 == 0,
    "multiples of 10": lambda n: n % 10 == 0,
}


@dataclass(frozen=True)
class Hypothesis:
    name: str
    family: str
    members: frozenset

    @property
    def size(self):
        return len(self.members)


def build_hypotheses() -> list[Hypothesis]:
    hyps = []
    for name, rule in MATHY.items():
        hyps.append(Hypothesis(name, "mathy", frozenset(n for n in DOMAIN if rule(n))))
    for a in DOMAIN:
        for b in range(a, 101):
            hyps.append(Hypothesis(f"[{a},{b}]", "interval", frozenset(range(a, b + 1))))
    return hyps


HYPOTHESES = build_hypotheses()

PRIORS = {
    "balanced": {"mathy": 0.5, "interval": 0.5},
    "mathy-biased": {"mathy": 0.95, "interval": 0.05},
    "interval-only": {"mathy": 0.0, "interval": 1.0},
    "mathy-only": {"mathy": 1.0, "interval": 0.0},
}


def prior_over(family_weights: dict, hypotheses=HYPOTHESES) -> dict:
    """Spread each family's weight evenly across its hypotheses."""
    counts = {}
    for h in hypotheses:
        counts[h.family] = counts.get(h.family, 0) + 1
    return {h: family_weights.get(h.family, 0.0) / counts[h.family] for h in hypotheses}


def infer(data, prior: dict, sampling: str = "strong") -> dict:
    """Posterior over hypotheses. sampling is 'strong' (size principle) or 'weak'."""
    data = list(data)
    post = {}
    for h, p in prior.items():
        if all(x in h.members for x in data):
            like = (1 / h.size) ** len(data) if sampling == "strong" else 1.0
        else:
            like = 0.0
        post[h] = p * like
    total = sum(post.values())
    if total == 0:
        raise ValueError("no hypothesis is consistent with the data under this prior")
    return {h: p / total for h, p in post.items()}


def generalise(posterior_: dict) -> dict:
    """P(x belongs to the concept | data), averaging over hypotheses."""
    return {x: sum(p for h, p in posterior_.items() if x in h.members) for x in DOMAIN}


def top(posterior_: dict, k: int = 5):
    return sorted(((h.name, p) for h, p in posterior_.items()), key=lambda t: -t[1])[:k]
