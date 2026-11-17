"""A small forward-chaining inference engine over Horn clauses, with proof traces.

Leibniz imagined a calculus ratiocinator: reasoning as calculation. This is a
tiny one. Facts are ground atoms, rules have the form

    human(?x) -> mortal(?x)
    parent(?a, ?b), parent(?b, ?c) -> grandparent(?a, ?c)

Variables start with '?'. Every variable in a rule head must also occur in its
body. Running the engine derives everything that follows and remembers why, so
any conclusion can be explained back to the facts it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ATOM = re.compile(r"\s*(\w+)\s*\(([^)]*)\)\s*")
_ATOM_FIND = re.compile(r"\w+\s*\([^)]*\)")


def parse_atom(text: str) -> tuple:
    m = _ATOM.fullmatch(text)
    if not m:
        raise ValueError(f"bad atom: {text!r}")
    args = tuple(a.strip() for a in m.group(2).split(",") if a.strip())
    return (m.group(1),) + args


def fmt(atom: tuple) -> str:
    return f"{atom[0]}({', '.join(atom[1:])})"


def is_var(term: str) -> bool:
    return term.startswith("?")


def match(pattern: tuple, fact: tuple, binding: dict):
    """One-way match of a pattern against a ground fact. Returns new binding or None."""
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    b = dict(binding)
    for p, f in zip(pattern[1:], fact[1:]):
        if is_var(p):
            if p in b and b[p] != f:
                return None
            b[p] = f
        elif p != f:
            return None
    return b


def substitute(atom: tuple, binding: dict) -> tuple:
    return (atom[0],) + tuple(binding[t] if is_var(t) else t for t in atom[1:])


@dataclass(frozen=True)
class Rule:
    head: tuple
    body: tuple
    text: str


class KnowledgeBase:
    def __init__(self):
        self.rules: list[Rule] = []
        self.facts: dict[tuple, tuple | None] = {}  # atom -> None (given) or (rule, support)

    def tell(self, text: str) -> "KnowledgeBase":
        if "->" in text:
            body_s, head_s = text.split("->", 1)
            body = tuple(parse_atom(a) for a in _ATOM_FIND.findall(body_s))
            head = parse_atom(head_s)
            body_vars = {t for a in body for t in a[1:] if is_var(t)}
            if not {t for t in head[1:] if is_var(t)} <= body_vars:
                raise ValueError(f"unsafe rule, head variable not in body: {text!r}")
            self.rules.append(Rule(head, body, text.strip()))
        else:
            atom = parse_atom(text)
            if any(is_var(t) for t in atom[1:]):
                raise ValueError(f"facts must be ground: {text!r}")
            self.facts.setdefault(atom, None)
        return self

    def _solve(self, body, binding, support):
        if not body:
            yield binding, support
            return
        first, rest = body[0], body[1:]
        for fact in list(self.facts):
            b = match(first, fact, binding)
            if b is not None:
                yield from self._solve(rest, b, support + [fact])

    def run(self) -> int:
        """Forward chain to a fixpoint. Returns the number of passes."""
        passes, changed = 0, True
        while changed:
            changed, passes = False, passes + 1
            for rule in self.rules:
                for binding, support in list(self._solve(rule.body, {}, [])):
                    new = substitute(rule.head, binding)
                    if new not in self.facts:
                        self.facts[new] = (rule, tuple(support))
                        changed = True
        return passes

    def ask(self, text: str) -> list[dict]:
        """Return all variable bindings making the pattern a known fact."""
        pattern = parse_atom(text)
        out = []
        for fact in self.facts:
            b = match(pattern, fact, {})
            if b is not None:
                out.append(b)
        return sorted(out, key=lambda d: sorted(d.items()))

    def explain(self, text: str) -> str:
        """A proof tree for a ground atom, down to the given facts."""
        lines: list[str] = []

        def go(atom, depth):
            pad = "  " * depth
            if atom not in self.facts:
                lines.append(f"{pad}{fmt(atom)}  [not derivable]")
                return
            just = self.facts[atom]
            if just is None:
                lines.append(f"{pad}{fmt(atom)}  [given]")
                return
            rule, support = just
            lines.append(f"{pad}{fmt(atom)}  [by {rule.text}]")
            for s in support:
                go(s, depth + 1)

        go(parse_atom(text), 0)
        return "\n".join(lines)
