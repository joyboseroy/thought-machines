"""Genetic programming: evolve the structure of a program, not just its parameters.

Trees are nested tuples, ('+', left, right), with terminals as variable names or
numbers. The engine is language-agnostic: give it a function set, a terminal set
and a fitness function (lower is better). Gradient descent tunes numbers inside a
fixed form. Evolution can change the form itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Language:
    functions: dict  # name -> (arity, callable)
    terminals: list  # variable names (str) and/or constants


@dataclass
class GPResult:
    tree: object
    error: float
    history: list
    generations: int


def random_tree(lang: Language, rng: random.Random, depth: int, full: bool = False):
    if depth <= 0 or (not full and rng.random() < 0.25):
        return rng.choice(lang.terminals)
    name = rng.choice(list(lang.functions))
    arity = lang.functions[name][0]
    return (name,) + tuple(random_tree(lang, rng, depth - 1, full) for _ in range(arity))


def evaluate(tree, lang: Language, env: dict):
    if isinstance(tree, tuple):
        fn = lang.functions[tree[0]][1]
        return fn(*[evaluate(c, lang, env) for c in tree[1:]])
    if isinstance(tree, str):
        return env[tree]
    return tree


def size(tree) -> int:
    return 1 + sum(size(c) for c in tree[1:]) if isinstance(tree, tuple) else 1


def depth(tree) -> int:
    return 1 + max(depth(c) for c in tree[1:]) if isinstance(tree, tuple) else 0


def paths(tree, prefix=()):
    yield prefix
    if isinstance(tree, tuple):
        for i, c in enumerate(tree[1:], start=1):
            yield from paths(c, prefix + (i,))


def get(tree, path):
    for i in path:
        tree = tree[i]
    return tree


def replace(tree, path, new):
    if not path:
        return new
    i = path[0]
    return tree[:i] + (replace(tree[i], path[1:], new),) + tree[i + 1:]


def crossover(a, b, rng):
    pa = rng.choice(list(paths(a)))
    pb = rng.choice(list(paths(b)))
    return replace(a, pa, get(b, pb))


def mutate(tree, lang, rng):
    return replace(tree, rng.choice(list(paths(tree))), random_tree(lang, rng, 2))


def to_str(tree) -> str:
    if not isinstance(tree, tuple):
        return str(tree)
    op, args = tree[0], [to_str(c) for c in tree[1:]]
    if len(args) == 2:
        return f"({args[0]} {op} {args[1]})"
    return f"{op}({', '.join(args)})"


def evolve(lang: Language, fitness, pop_size: int = 200, generations: int = 40, max_depth: int = 6,
           init_depth: int = 3, tournament: int = 5, p_cross: float = 0.8, p_mut: float = 0.2,
           parsimony: float = 0.0, target_error: float = 0.0, elite: int = 2, seed: int = 0) -> GPResult:
    rng = random.Random(seed)
    pop = [random_tree(lang, rng, rng.randint(1, init_depth), full=(i % 2 == 0)) for i in range(pop_size)]
    cache: dict = {}

    def err(t):
        if t not in cache:
            cache[t] = fitness(t)
        return cache[t]

    history, best, best_err = [], None, float("inf")
    gens = 0
    for gens in range(generations + 1):
        errs = [err(t) for t in pop]
        scores = [e + parsimony * size(t) for e, t in zip(errs, pop)]
        order = sorted(range(len(pop)), key=scores.__getitem__)
        top = order[0]
        if errs[top] < best_err or (errs[top] == best_err and size(pop[top]) < size(best)):
            best, best_err = pop[top], errs[top]
        history.append(best_err)
        if best_err <= target_error or gens == generations:
            break

        def pick():
            cand = rng.sample(range(len(pop)), min(tournament, len(pop)))
            return pop[min(cand, key=scores.__getitem__)]

        nxt = [pop[i] for i in order[:elite]]
        while len(nxt) < pop_size:
            child = pick()
            if rng.random() < p_cross:
                child = crossover(child, pick(), rng)
            if rng.random() < p_mut:
                child = mutate(child, lang, rng)
            nxt.append(child if depth(child) <= max_depth else pick())
        pop = nxt
    return GPResult(best, best_err, history, gens)


def regression_language(constants=(1, 2, 3)) -> Language:
    return Language(
        functions={"+": (2, lambda a, b: a + b), "-": (2, lambda a, b: a - b),
                   "*": (2, lambda a, b: a * b)},
        terminals=["x", *constants],
    )


def boolean_language(names, full_mask: int) -> Language:
    """Boolean expressions evaluated on bit-vectors, one bit per training example."""
    return Language(
        functions={"and": (2, lambda a, b: a & b), "or": (2, lambda a, b: a | b),
                   "not": (1, lambda a: a ^ full_mask)},
        terminals=list(names),
    )
