import random

from .decision_tree import DecisionTree
from .ilp import closed_world_negatives, learn_program


def main():
    print("== Decision tree: learn a rule from examples ==")
    rng = random.Random(0)
    X = [tuple(rng.randint(0, 1) for _ in range(5)) for _ in range(60)]
    y = [int((x[0] and x[1]) or x[2]) for x in X]
    tree = DecisionTree().fit(X, y)
    print("hidden rule: (x0 AND x1) OR x2, with x3 and x4 as distractors")
    print(f"tree has {tree.size()} nodes, training accuracy "
          f"{sum(tree.predict(x) == t for x, t in zip(X, y)) / len(X):.2f}")
    print(tree.to_text())

    print("\n== Inverse deduction: discover a rule from facts ==")
    background = [
        "parent(abe, homer)", "parent(mona, homer)", "parent(homer, bart)",
        "parent(homer, lisa)", "parent(marge, bart)", "parent(marge, lisa)",
        "parent(clancy, marge)", "parent(jackie, marge)", "parent(lisa, maggie)",
    ]
    positives = ["grandparent(abe, bart)", "grandparent(abe, lisa)", "grandparent(mona, bart)",
                 "grandparent(mona, lisa)", "grandparent(clancy, bart)", "grandparent(clancy, lisa)",
                 "grandparent(jackie, bart)", "grandparent(jackie, lisa)", "grandparent(homer, maggie)",
                 "grandparent(marge, maggie)"]
    people = ["abe", "mona", "homer", "marge", "bart", "lisa", "clancy", "jackie", "maggie"]
    negatives = closed_world_negatives(people, positives, "grandparent")
    for r in learn_program(background, positives, negatives, "grandparent"):
        print(f"learned: {r.rule}   (covers {r.new_positives} examples)")

    print("\nA concept that needs two rules (a disjunction):")
    background = ["mother(ann, bob)", "father(carl, bob)", "mother(ann, dee)", "father(carl, dee)",
                  "mother(eve, ann)", "father(fred, ann)", "likes(bob, dee)"]
    positives = ["parent(ann, bob)", "parent(carl, bob)", "parent(ann, dee)", "parent(carl, dee)",
                 "parent(eve, ann)", "parent(fred, ann)"]
    names = ["ann", "bob", "carl", "dee", "eve", "fred"]
    negatives = closed_world_negatives(names, positives, "parent")
    for r in learn_program(background, positives, negatives, "parent"):
        print(f"learned: {r.rule}   (covers {r.new_positives} new examples)")


if __name__ == "__main__":
    main()
