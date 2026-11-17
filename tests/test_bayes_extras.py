import math
import unittest

from thought_machines.bayes.mcmc import coin_log_posterior, metropolis
from thought_machines.bayes.naive_bayes import NaiveBayes
from thought_machines.bayes.network import BayesNet, burglary_alarm


class NetworkTests(unittest.TestCase):
    def test_textbook_alarm_numbers(self):
        net = burglary_alarm()
        self.assertAlmostEqual(net.prob({"Burglary": True},
                                        {"JohnCalls": True, "MaryCalls": True}), 0.284, places=3)

    def test_explaining_away(self):
        net = burglary_alarm()
        given_alarm = net.prob({"Burglary": True}, {"Alarm": True})
        given_alarm_and_quake = net.prob({"Burglary": True}, {"Alarm": True, "Earthquake": True})
        self.assertLess(given_alarm_and_quake, given_alarm / 10)

    def test_marginals_and_complements(self):
        net = burglary_alarm()
        self.assertAlmostEqual(net.prob({"Burglary": True}), 0.001)
        self.assertAlmostEqual(net.prob({"Alarm": True}) + net.prob({"Alarm": False}), 1.0)
        self.assertAlmostEqual(net.prob({"Burglary": True}, {"Burglary": True}), 1.0)
        self.assertEqual(net.prob({"Burglary": True}, {"Burglary": False}), 0.0)

    def test_joint_sums_to_one(self):
        net = burglary_alarm()
        from itertools import product
        total = sum(net.joint(dict(zip(net.order, vals)))
                    for vals in product((False, True), repeat=len(net.order)))
        self.assertAlmostEqual(total, 1.0)

    def test_do_cuts_incoming_edges_and_leaves_original_alone(self):
        net = burglary_alarm()
        cut = net.do("Alarm", True)
        self.assertAlmostEqual(cut.prob({"Alarm": True}), 1.0)
        self.assertAlmostEqual(cut.prob({"Burglary": True}), 0.001)  # no longer informed by the alarm
        self.assertAlmostEqual(net.prob({"Alarm": True}), net.prob({"Alarm": True}))
        self.assertLess(net.prob({"Alarm": True}), 0.01)

    def test_bad_inputs_raise(self):
        net = BayesNet().add("A", (), {(): 0.5})
        with self.assertRaises(ValueError):
            net.add("B", ("A",), {(True,): 0.9})  # missing row
        with self.assertRaises(ValueError):
            net.add("C", ("Z",), {(True,): 0.9, (False,): 0.1})  # parent not added
        with self.assertRaises(KeyError):
            net.prob({"Nope": True})


class NaiveBayesTests(unittest.TestCase):
    X = [(a, b) for a in (0, 1) for b in (0, 1)] * 10

    def test_and_is_learned(self):
        nb = NaiveBayes().fit(self.X, [a & b for a, b in self.X])
        self.assertEqual(nb.predict_all(self.X), [a & b for a, b in self.X])

    def test_xor_is_invisible_to_independence_assumptions(self):
        y = [a ^ b for a, b in self.X]
        nb = NaiveBayes().fit(self.X, y)
        acc = sum(p == t for p, t in zip(nb.predict_all(self.X), y)) / len(y)
        self.assertEqual(acc, 0.5)

    def test_probabilities_normalise(self):
        nb = NaiveBayes().fit(self.X, [a & b for a, b in self.X])
        self.assertAlmostEqual(sum(nb.predict_proba((1, 1)).values()), 1.0)
        self.assertGreater(nb.predict_proba((9, 9))[0], 0)  # unseen values are smoothed, not fatal


class MCMCTests(unittest.TestCase):
    def test_posterior_mean_matches_the_exact_beta(self):
        samples, acc = metropolis(coin_log_posterior(7, 3), 0.5, step=0.2)
        self.assertAlmostEqual(sum(samples) / len(samples), 8 / 12, delta=0.02)
        self.assertTrue(0.2 < acc < 0.9)

    def test_posterior_variance_matches_the_exact_beta(self):
        samples, _ = metropolis(coin_log_posterior(7, 3), 0.5, steps=40000, step=0.2)
        m = sum(samples) / len(samples)
        var = sum((s - m) ** 2 for s in samples) / len(samples)
        exact = (8 * 4) / (12 ** 2 * 13)
        self.assertAlmostEqual(var, exact, delta=0.004)

    def test_support_is_respected(self):
        samples, _ = metropolis(coin_log_posterior(1, 1), 0.5, step=0.5)
        self.assertTrue(all(0 < s < 1 for s in samples))
        self.assertTrue(math.isinf(coin_log_posterior(1, 1)(1.5)))


if __name__ == "__main__":
    unittest.main()
