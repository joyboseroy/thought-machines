import unittest

from thought_machines.bayes.inference import posterior, sequential_update, update
from thought_machines.bayes.number_game import PRIORS, generalise, infer, prior_over


class BayesTests(unittest.TestCase):
    def test_textbook_medical_test(self):
        self.assertAlmostEqual(posterior(0.01, 0.95, 0.05), 0.0095 / (0.0095 + 0.0495))

    def test_sequential_equals_batch(self):
        # two positives in a row, in either order, give the same belief
        a = sequential_update(0.01, 0.9, 0.1, [True, True])[-1]
        b = sequential_update(0.01, 0.9, 0.1, [True, False, True, True, False])[-1]
        self.assertGreater(a, 0.01)
        self.assertLess(b, a)

    def test_evidence_order_does_not_matter(self):
        x = sequential_update(0.2, 0.8, 0.3, [True, False, True])[-1]
        y = sequential_update(0.2, 0.8, 0.3, [False, True, True])[-1]
        self.assertAlmostEqual(x, y)

    def test_general_update(self):
        post = update({"a": 0.5, "b": 0.5}, lambda h: 0.9 if h == "a" else 0.1)
        self.assertAlmostEqual(post["a"], 0.9)


class NumberGameTests(unittest.TestCase):
    def setUp(self):
        self.balanced = prior_over(PRIORS["balanced"])

    def test_posterior_sums_to_one(self):
        self.assertAlmostEqual(sum(infer([16], self.balanced).values()), 1.0)

    def test_size_principle_picks_powers_of_two(self):
        g = generalise(infer([16, 8, 2, 64], self.balanced))
        self.assertGreater(g[32], 0.95)
        self.assertLess(g[30], 0.05)

    def test_clustered_data_generalises_as_interval(self):
        g = generalise(infer([16, 23, 19, 20], self.balanced))
        self.assertGreater(g[21], 0.9)
        self.assertLess(g[60], 0.3)

    def test_prior_changes_conclusions_from_identical_data(self):
        mathy = generalise(infer([16], prior_over(PRIORS["mathy-only"])))
        interval = generalise(infer([16], prior_over(PRIORS["interval-only"])))
        self.assertEqual(mathy[17], 0.0)
        self.assertGreater(interval[17], 0.3)

    def test_weak_sampling_loses_the_size_principle(self):
        # 30 is an even number inside [2, 64] but not a power of two. Only the size
        # principle, which prefers the small consistent concept, rules it out.
        strong = generalise(infer([16, 8, 2, 64], self.balanced, "strong"))
        weak = generalise(infer([16, 8, 2, 64], self.balanced, "weak"))
        self.assertLess(strong[30], 0.01)
        self.assertGreater(weak[30], 0.3)

    def test_impossible_data_raises(self):
        with self.assertRaises(ValueError):
            infer([16, 23, 19, 20], prior_over(PRIORS["mathy-only"]))


if __name__ == "__main__":
    unittest.main()
