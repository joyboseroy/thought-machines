"""A plain genetic algorithm over bit strings."""

from __future__ import annotations

import random


def genetic_algorithm(fitness, n_bits: int, pop_size: int = 60, generations: int = 80,
                      p_mut: float | None = None, tournament: int = 3, elite: int = 1, seed: int = 0):
    """Maximise `fitness(bits)`. Returns (best_bits, best_fitness, history)."""
    rng = random.Random(seed)
    p_mut = 1.0 / n_bits if p_mut is None else p_mut
    pop = [[rng.randint(0, 1) for _ in range(n_bits)] for _ in range(pop_size)]
    history = []
    best, best_f = None, float("-inf")
    for _ in range(generations + 1):
        fits = [fitness(ind) for ind in pop]
        i = max(range(pop_size), key=fits.__getitem__)
        if fits[i] > best_f:
            best, best_f = list(pop[i]), fits[i]
        history.append(best_f)

        def pick():
            return pop[max(rng.sample(range(pop_size), tournament), key=fits.__getitem__)]

        order = sorted(range(pop_size), key=fits.__getitem__, reverse=True)
        nxt = [list(pop[j]) for j in order[:elite]]
        while len(nxt) < pop_size:
            a, b = pick(), pick()
            cut = rng.randrange(1, n_bits)
            child = a[:cut] + b[cut:]
            nxt.append([1 - g if rng.random() < p_mut else g for g in child])
        pop = nxt
    return best, best_f, history
