"""Resource rationality: when is a worse algorithm the better decision?

Task: choose the best of N options whose values are unknown draws from a standard
normal. Looking at an option costs `cost` utility. Three kinds of agent:

  * ideal:     looks at all N, always finds the best, pays for N looks.
  * bounded-k: looks at k options chosen at random, takes the best seen.
  * satisficer: looks in order and stops at the first option above a threshold.

Net utility = value of the chosen option minus cost times looks. Expected best
of k draws grows only like sqrt(2 ln k), while cost grows like k, so past some
point thinking more is a loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Row:
    name: str
    looks: float
    value: float
    net: float
    regret: float


def default_budgets(n: int):
    ks, k = [], 1
    while k < n:
        ks.append(k)
        k *= 2
    ks.append(n)
    return ks


def compare(n_options: int = 1000, trials: int = 2000, cost: float = 0.01,
            budgets=None, threshold: float = 1.5, seed: int = 0) -> list[Row]:
    """Monte Carlo comparison. The same random worlds are used for every agent."""
    budgets = sorted(set(budgets or default_budgets(n_options)))
    rng = random.Random(seed)
    value = {k: 0.0 for k in budgets}
    regret = {k: 0.0 for k in budgets}
    sat_value = sat_looks = sat_regret = 0.0
    for _ in range(trials):
        world = [rng.gauss(0, 1) for _ in range(n_options)]
        best = max(world)
        running, bi = float("-inf"), 0
        sat_choice = sat_n = None
        for i, v in enumerate(world, start=1):
            running = max(running, v)
            if sat_choice is None and v >= threshold:
                sat_choice, sat_n = v, i
            if bi < len(budgets) and i == budgets[bi]:
                value[budgets[bi]] += running
                regret[budgets[bi]] += best - running
                bi += 1
        if sat_choice is None:
            sat_choice, sat_n = best, n_options
        sat_value += sat_choice
        sat_looks += sat_n
        sat_regret += best - sat_choice
    rows = []
    for k in budgets:
        name = f"ideal (k={k})" if k == n_options else f"bounded k={k}"
        v = value[k] / trials
        rows.append(Row(name, k, v, v - cost * k, regret[k] / trials))
    v, l = sat_value / trials, sat_looks / trials
    rows.append(Row(f"satisficer >= {threshold}", l, v, v - cost * l, sat_regret / trials))
    return rows


def best_row(rows: list[Row]) -> Row:
    return max(rows, key=lambda r: r.net)
