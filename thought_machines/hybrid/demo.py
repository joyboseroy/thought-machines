import random

from ..symbolist.decision_tree import DecisionTree
from .rule_logit import RuleLogit


def _data(rng, n, noise):
    X = [tuple(rng.randint(0, 1) for _ in range(8)) for _ in range(n)]
    clean = [int((x[0] and x[1]) or x[2]) for x in X]
    return X, [t ^ (rng.random() < noise) for t in clean], clean


def main():
    print("Hidden rule (x0 AND x1) OR x2 over 8 binary features, 5 distractors.")
    print("Plain tree versus RuleLogit (same tree, probabilistic rule weights), 40 training examples.")
    print(f"{'label noise':>12} {'tree':>8} {'RuleLogit':>10}")
    for noise in (0.0, 0.1, 0.2, 0.3):
        t_acc = r_acc = 0.0
        reps = 20
        for rep in range(reps):
            rng = random.Random(rep)
            Xtr, ytr, _ = _data(rng, 40, noise)
            Xte, _, yte = _data(rng, 300, 0.0)
            tree = DecisionTree().fit(Xtr, ytr)
            rl = RuleLogit().fit(Xtr, ytr)
            t_acc += sum(p == t for p, t in zip(tree.predict_all(Xte), yte)) / len(yte)
            r_acc += sum(p == t for p, t in zip(rl.predict_all(Xte), yte)) / len(yte)
        print(f"{noise:>12.1f} {t_acc / reps:>8.2f} {r_acc / reps:>10.2f}")


if __name__ == "__main__":
    main()
