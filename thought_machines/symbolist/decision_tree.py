"""ID3 decision trees: the symbolist tribe learning rules from examples.

Inputs are tuples of discrete feature values. Each split picks the feature with
the highest information gain. The learned tree can be printed as readable rules,
which is its main selling point against the other tribes.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


def entropy(labels) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values())


@dataclass
class Node:
    id: int
    label: object  # majority label of the training examples that reached this node
    n: int
    feature: int | None = None
    children: dict = field(default_factory=dict)


class DecisionTree:
    def __init__(self, max_depth: int | None = None, min_gain: float = 1e-9):
        self.max_depth, self.min_gain = max_depth, min_gain
        self.root: Node | None = None
        self._count = 0

    def fit(self, X, y):
        self._count = 0
        self.root = self._build([tuple(x) for x in X], list(y), list(range(len(X[0]))), 0)
        return self

    def _new(self, y) -> Node:
        node = Node(self._count, Counter(y).most_common(1)[0][0], len(y))
        self._count += 1
        return node

    def _build(self, X, y, feats, depth) -> Node:
        node = self._new(y)
        if len(set(y)) == 1 or not feats or (self.max_depth is not None and depth >= self.max_depth):
            return node
        base, best = entropy(y), None
        for f in feats:
            groups: dict = {}
            for xi, yi in zip(X, y):
                groups.setdefault(xi[f], []).append(yi)
            gain = base - sum(len(g) / len(y) * entropy(g) for g in groups.values())
            if best is None or gain > best[0] + 1e-12:
                best = (gain, f)
        if best[0] <= self.min_gain:
            return node
        f = best[1]
        node.feature = f
        rest = [g for g in feats if g != f]
        for v in sorted({x[f] for x in X}):
            idx = [i for i, x in enumerate(X) if x[f] == v]
            node.children[v] = self._build([X[i] for i in idx], [y[i] for i in idx], rest, depth + 1)
        return node

    def _walk(self, x) -> Node:
        node = self.root
        while node.feature is not None:
            nxt = node.children.get(x[node.feature])
            if nxt is None:
                break  # unseen value: fall back to this node's majority label
            node = nxt
        return node

    def predict(self, x):
        return self._walk(x).label

    def apply(self, x) -> int:
        """Id of the node an input ends up in."""
        return self._walk(x).id

    def predict_all(self, X):
        return [self.predict(x) for x in X]

    def size(self) -> int:
        def count(n):
            return 1 + sum(count(c) for c in n.children.values())
        return count(self.root)

    def n_nodes(self) -> int:
        return self._count

    def rules(self, names=None):
        """One (conditions, label) pair per leaf. Conditions are (feature, value) pairs."""
        out = []

        def go(node, conds):
            if node.feature is None:
                out.append((conds, node.label))
                return
            for v, child in node.children.items():
                go(child, conds + [(node.feature, v)])

        go(self.root, [])
        return out

    def to_text(self, names=None) -> str:
        lines = []
        for conds, label in self.rules():
            body = " AND ".join(f"{(names[f] if names else f'x{f}')}={v}" for f, v in conds) or "TRUE"
            lines.append(f"IF {body} THEN {label}")
        return "\n".join(lines)
