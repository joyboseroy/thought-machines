import unittest
from itertools import product

from thought_machines.analogizer.kernel import KernelSVM, accuracy, linear, make_circles, rbf
from thought_machines.analogizer.knn import KNN, hamming
from thought_machines.evolutionary.ga import genetic_algorithm
from thought_machines.evolutionary.gp import evaluate, evolve, regression_language, to_str
from thought_machines.hybrid.rule_logit import RuleLogit
from thought_machines.logic.engine import KnowledgeBase
from thought_machines.symbolist.decision_tree import DecisionTree
from thought_machines.symbolist.ilp import closed_world_negatives, learn_program


def hidden_rule_data():
    X = list(product((0, 1), repeat=5))
    return X, [int((x[0] and x[1]) or x[2]) for x in X]  # x3 and x4 are distractors


class SymbolistTests(unittest.TestCase):
    def test_tree_recovers_hidden_rule_exactly(self):
        X, y = hidden_rule_data()
        tree = DecisionTree().fit(X, y)
        self.assertEqual(tree.predict_all(X), y)

    def test_tree_ignores_distractor_features(self):
        X, y = hidden_rule_data()
        used = {f for conds, _ in DecisionTree().fit(X, y).rules() for f, _ in conds}
        self.assertEqual(used, {0, 1, 2})

    def test_rules_text_mentions_every_leaf(self):
        X, y = hidden_rule_data()
        tree = DecisionTree().fit(X, y)
        self.assertEqual(len(tree.to_text().splitlines()), len(tree.rules()))

    def test_unseen_value_falls_back_to_majority(self):
        tree = DecisionTree().fit([(0,), (0,), (1,)], [0, 0, 1])
        self.assertIn(tree.predict((7,)), (0, 1))

    def test_ilp_discovers_grandparent_and_engine_reproduces_it(self):
        people = ["alice", "bob", "carol", "dave", "eve", "frank"]
        bg = ["parent(alice, bob)", "parent(bob, carol)", "parent(carol, dave)",
              "parent(alice, eve)", "parent(eve, frank)"]
        pos = ["grandparent(alice, carol)", "grandparent(bob, dave)", "grandparent(alice, frank)"]
        neg = closed_world_negatives(people, pos, "grandparent")
        program = learn_program(bg, pos, neg, "grandparent")
        self.assertEqual(len(program), 1)
        kb = KnowledgeBase()
        for fact in bg:
            kb.tell(fact)
        for item in program:
            kb.tell(item.rule)
        kb.run()
        derived = {(b["?x"], b["?y"]) for b in kb.ask("grandparent(?x, ?y)")}
        self.assertEqual(derived, {("alice", "carol"), ("bob", "dave"), ("alice", "frank")})


class AnalogizerTests(unittest.TestCase):
    def test_knn_hamming_memorises_and_votes(self):
        knn = KNN(1, distance=hamming).fit([(0, 0), (1, 1)], ["a", "b"])
        self.assertEqual(knn.predict((0, 1)), "a")  # tie in distance, first stored wins
        self.assertEqual(knn.predict((1, 1)), "b")

    def test_linear_kernel_cannot_separate_rings_but_rbf_can(self):
        Xtr, ytr = make_circles(120, seed=1)
        Xte, yte = make_circles(200, seed=2)
        lin = KernelSVM(linear, epochs=20).fit(Xtr, ytr)
        non = KernelSVM(rbf(2.0), epochs=20).fit(Xtr, ytr)
        self.assertLess(accuracy(lin, Xte, yte), 0.7)
        self.assertGreater(accuracy(non, Xte, yte), 0.95)

    def test_xor_needs_a_nonlinear_kernel(self):
        X, y = [(-1, -1), (-1, 1), (1, -1), (1, 1)], [0, 1, 1, 0]
        self.assertLess(accuracy(KernelSVM(linear).fit(X, y), X, y), 1.0)
        self.assertEqual(accuracy(KernelSVM(rbf(1.0)).fit(X, y), X, y), 1.0)


class EvolutionaryTests(unittest.TestCase):
    def test_ga_solves_onemax(self):
        best, fit, history = genetic_algorithm(sum, 40, seed=0)
        self.assertEqual(fit, 40)
        self.assertEqual(history, sorted(history))  # best-so-far never decreases

    def test_gp_discovers_the_structure_and_extrapolates(self):
        X = [x / 2 for x in range(-10, 11)]
        Y = [x * x + 3 * x + 1 for x in X]
        lang = regression_language()

        def error(tree):
            return sum((evaluate(tree, lang, {"x": x}) - y) ** 2 for x, y in zip(X, Y)) / len(X)

        for seed in range(3):
            r = evolve(lang, error, seed=seed, target_error=1e-9, parsimony=1e-4)
            self.assertEqual(r.error, 0.0, to_str(r.tree))
            # trained on [-5, 5]; the discovered form is right far outside it
            self.assertEqual(evaluate(r.tree, lang, {"x": 20.0}), 20 * 20 + 60 + 1)

    def test_gp_is_deterministic_for_a_seed(self):
        lang = regression_language()
        f = lambda t: abs(evaluate(t, lang, {"x": 2.0}) - 11)
        a, b = evolve(lang, f, seed=3, generations=10), evolve(lang, f, seed=3, generations=10)
        self.assertEqual((a.tree, a.error), (b.tree, b.error))


class HybridTests(unittest.TestCase):
    def test_rulelogit_matches_tree_on_clean_data_and_gives_probabilities(self):
        X, y = hidden_rule_data()
        rl = RuleLogit().fit(X, y)
        self.assertGreaterEqual(sum(p == t for p, t in zip(rl.predict_all(X), y)) / len(y), 0.95)
        self.assertTrue(all(0.0 <= rl.proba(x) <= 1.0 for x in X))
        self.assertEqual(rl.n_rules(), len(rl.tree.rules()))  # one weight per leaf of the tree
        # rules backed by consistent evidence end up confident in the right direction
        self.assertGreater(rl.proba((1, 1, 0, 0, 0)), 0.5)
        self.assertLess(rl.proba((0, 0, 0, 1, 1)), 0.5)


if __name__ == "__main__":
    unittest.main()
