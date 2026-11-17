"""One adapter per learner, all with the same interface.

    learner = make(name, seed)
    learner.fit(X, y)
    learner.predict_all(X) -> list of 0/1
    learner.size() -> a rough model size (nodes, rules, parameters), or None

Hyperparameters are fixed across tasks. Nothing is tuned per task.
"""

from __future__ import annotations

from ..analogizer.knn import KNN, hamming
from ..bayes.naive_bayes import NaiveBayes
from ..evolutionary import gp
from ..hybrid.rule_logit import RuleLogit
from ..neural.mlp import MLP
from ..symbolist.decision_tree import DecisionTree

TRIBES = {
    "tree": "symbolist",
    "naive_bayes": "bayesian",
    "knn": "analogizer",
    "mlp": "connectionist",
    "gp": "evolutionary",
    "rule_logit": "hybrid",
}


class TreeLearner:
    def __init__(self, seed=0):
        self.m = DecisionTree()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_all(self, X):
        return self.m.predict_all(X)

    def size(self):
        return self.m.size()


class NaiveBayesLearner:
    def __init__(self, seed=0):
        self.m = NaiveBayes()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_all(self, X):
        return self.m.predict_all(X)

    def size(self):
        return None


class KNNLearner:
    def __init__(self, seed=0):
        self.m = KNN(k=3, distance=hamming)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_all(self, X):
        return self.m.predict_all(X)

    def size(self):
        return None


class MLPLearner:
    def __init__(self, seed=0, hidden=6, epochs=200, lr=0.3):
        self.seed, self.hidden, self.epochs, self.lr = seed, hidden, epochs, lr

    def fit(self, X, y):
        self.m = MLP(len(X[0]), self.hidden, seed=self.seed)
        self.m.fit(list(zip(X, y)), epochs=self.epochs, lr=self.lr, seed=self.seed, log_every=10**9)
        return self

    def predict_all(self, X):
        return [self.m.predict(x) for x in X]

    def size(self):
        return self.hidden * (len(self.m.w1[0]) + 2) + 1


class GPLearner:
    """Evolves a boolean expression over AND, OR, NOT. Examples are packed into big integers."""

    def __init__(self, seed=0, pop=120, generations=30):
        self.seed, self.pop, self.generations = seed, pop, generations

    @staticmethod
    def _pack(X):
        n = len(X)
        names = [f"x{i}" for i in range(len(X[0]))]
        env = {nm: sum(x[i] << j for j, x in enumerate(X)) for i, nm in enumerate(names)}
        return names, env, (1 << n) - 1

    def fit(self, X, y):
        names, env, mask = self._pack(X)
        self.names = names
        target = sum(t << j for j, t in enumerate(y))
        lang = gp.boolean_language(names, mask)
        n = len(X)

        def fitness(tree):
            return bin(gp.evaluate(tree, lang, env) ^ target).count("1") / n

        r = gp.evolve(lang, fitness, pop_size=self.pop, generations=self.generations,
                      max_depth=6, parsimony=0.002, seed=self.seed)
        self.tree = r.tree
        return self

    def predict_all(self, X):
        _, env, mask = self._pack(X)
        out = gp.evaluate(self.tree, gp.boolean_language(self.names, mask), env)
        return [(out >> j) & 1 for j in range(len(X))]

    def size(self):
        return gp.size(self.tree)

    def expression(self):
        return gp.to_str(self.tree)


class RuleLogitLearner:
    def __init__(self, seed=0):
        self.m = RuleLogit()

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_all(self, X):
        return self.m.predict_all(X)

    def size(self):
        return self.m.n_rules()


LEARNERS = {
    "tree": TreeLearner,
    "naive_bayes": NaiveBayesLearner,
    "knn": KNNLearner,
    "mlp": MLPLearner,
    "gp": GPLearner,
    "rule_logit": RuleLogitLearner,
}


def make(name: str, seed: int = 0):
    return LEARNERS[name](seed=seed)
