"""Inductive logic programming: inverse deduction in miniature.

Deduction runs rules forward to get facts. Induction runs the other way: given
background facts and examples of a target relation, search for rules that, run
forward by the engine in `logic.engine`, reproduce every positive example and no
negative one. This is a small FOIL-flavoured covering learner, not a full ILP
system. Rule bodies are limited to a few literals over a few variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from ..logic.engine import KnowledgeBase, parse_atom

HEAD_VARS = ("?x", "?y", "?u")
EXTRA_VARS = ("?z", "?w")


@dataclass
class Learned:
    rule: str
    new_positives: int
    negatives: int


def predicates(background) -> dict:
    preds = {}
    for text in background:
        atom = parse_atom(text)
        preds[atom[0]] = len(atom) - 1
    return preds


def _lit(literal) -> str:
    return f"{literal[0]}({', '.join(literal[1:])})"


def _candidates(preds, target, arity, max_body, extra):
    head = (target,) + HEAD_VARS[:arity]
    variables = list(HEAD_VARS[:arity]) + list(EXTRA_VARS[:extra])
    literals = [(p,) + args for p, k in preds.items() if p != target
                for args in product(variables, repeat=k)]
    for size in range(1, max_body + 1):
        for body in combinations(literals, size):
            used = {t for lit in body for t in lit[1:]}
            if not set(HEAD_VARS[:arity]) <= used:
                continue  # unsafe: a head variable would be unbound
            yield head, body


def _derive(background, rule_text, target):
    kb = KnowledgeBase()
    for fact in background:
        kb.tell(fact)
    kb.tell(rule_text)
    kb.run()
    return {a for a in kb.facts if a[0] == target}


def learn_program(background, positives, negatives, target, arity=2, max_body=2, extra_vars=1):
    """Sequential covering. Returns a list of Learned rules (possibly empty)."""
    preds = predicates(background)
    pos = {parse_atom(p) for p in positives}
    neg = {parse_atom(n) for n in negatives}
    pool = []
    for head, body in _candidates(preds, target, arity, max_body, extra_vars):
        text = f"{', '.join(_lit(l) for l in body)} -> {_lit(head)}"
        derived = _derive(background, text, target)
        if derived & neg:
            continue  # inconsistent with a negative example
        if derived & pos:
            pool.append((text, len(body), derived & pos))
    program, uncovered = [], set(pos)
    while uncovered and pool:
        pool.sort(key=lambda r: (-len(r[2] & uncovered), r[1], r[0]))
        text, _, covers = pool[0]
        gain = len(covers & uncovered)
        if gain == 0:
            break
        program.append(Learned(text, gain, 0))
        uncovered -= covers
        pool = [r for r in pool if r[2] & uncovered]
    return program


def closed_world_negatives(constants, positives, target, arity=2):
    """Every tuple of constants that is not a positive example."""
    pos = {parse_atom(p) for p in positives}
    out = []
    for args in product(constants, repeat=arity):
        atom = (target,) + tuple(args)
        if atom not in pos:
            out.append(f"{target}({', '.join(args)})")
    return out
