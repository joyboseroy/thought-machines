"""Bayesian networks over binary variables, with exact inference by enumeration.

Each variable has a table giving P(var = True) for every combination of its
parents' values, in the order the parents were listed. The joint distribution is
the product of these tables. Intervening on a variable (Pearl's do operator) is
graph surgery: cut the variable off from its parents and pin it to a value.

    net.prob({"Burglary": True}, {"JohnCalls": True, "MaryCalls": True})
        is P(Burglary | JohnCalls, MaryCalls)
"""

from __future__ import annotations

import copy
from itertools import product


class BayesNet:
    def __init__(self):
        self.order: list[str] = []
        self.parents: dict[str, tuple] = {}
        self.cpt: dict[str, dict] = {}

    def add(self, name: str, parents=(), cpt=None) -> "BayesNet":
        """cpt maps a tuple of parent values to P(name = True). A bare float is allowed for roots."""
        parents = tuple(parents)
        if not isinstance(cpt, dict):
            cpt = {(): float(cpt)}
        for p in parents:
            if p not in self.parents:
                raise ValueError(f"parent {p!r} must be added before {name!r}")
        for combo in product((False, True), repeat=len(parents)):
            if combo not in cpt:
                raise ValueError(f"missing cpt row {combo} for {name!r}")
        self.order.append(name)
        self.parents[name], self.cpt[name] = parents, cpt
        return self

    def joint(self, assignment: dict) -> float:
        p = 1.0
        for v in self.order:
            p1 = self.cpt[v][tuple(bool(assignment[q]) for q in self.parents[v])]
            p *= p1 if assignment[v] else 1 - p1
        return p

    def _mass(self, fixed: dict) -> float:
        """Total probability of all worlds consistent with `fixed`."""
        free = [v for v in self.order if v not in fixed]
        return sum(self.joint({**fixed, **dict(zip(free, combo))})
                   for combo in product((False, True), repeat=len(free)))

    def prob(self, query: dict, evidence: dict | None = None) -> float:
        """P(query | evidence). With no evidence this is the marginal of the query."""
        evidence = evidence or {}
        for k, v in query.items():
            if k in evidence and bool(evidence[k]) != bool(v):
                return 0.0
        unknown = [k for k in {**query, **evidence} if k not in self.parents]
        if unknown:
            raise KeyError(f"unknown variable(s): {unknown}")
        denom = self._mass({k: bool(v) for k, v in evidence.items()})
        if denom == 0:
            raise ValueError("evidence has probability zero")
        both = {k: bool(v) for k, v in {**evidence, **query}.items()}
        return self._mass(both) / denom

    def do(self, var: str, value: bool) -> "BayesNet":
        """The network after do(var = value): var loses its parents and is fixed."""
        net = copy.deepcopy(self)
        net.parents[var] = ()
        net.cpt[var] = {(): 1.0 if value else 0.0}
        return net


def burglary_alarm() -> BayesNet:
    """The standard alarm network (Russell and Norvig). P(Burglary | John and Mary call) is about 0.284."""
    net = BayesNet()
    net.add("Burglary", (), {(): 0.001})
    net.add("Earthquake", (), {(): 0.002})
    net.add("Alarm", ("Burglary", "Earthquake"), {
        (True, True): 0.95, (True, False): 0.94, (False, True): 0.29, (False, False): 0.001})
    net.add("JohnCalls", ("Alarm",), {(True,): 0.90, (False,): 0.05})
    net.add("MaryCalls", ("Alarm",), {(True,): 0.70, (False,): 0.01})
    return net
