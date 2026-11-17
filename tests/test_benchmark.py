import csv
import os
import tempfile
import unittest

from thought_machines.benchmark.learners import LEARNERS, TRIBES, make
from thought_machines.benchmark.runner import mean_acc, report, run, write_csv
from thought_machines.benchmark.tasks import ALL_INPUTS, TASKS, make_split
import random


class TaskTests(unittest.TestCase):
    def test_split_is_exact_and_disjoint(self):
        Xtr, ytr, Xte, yte = make_split("dnf", 40, 0.0, random.Random(0))
        self.assertEqual(len(Xtr) + len(Xte), len(ALL_INPUTS))
        self.assertFalse(set(Xtr) & set(Xte))
        self.assertEqual(yte, [TASKS["dnf"](x) for x in Xte])
        self.assertEqual(ytr, [TASKS["dnf"](x) for x in Xtr])  # no noise requested

    def test_noise_flips_roughly_the_requested_fraction(self):
        Xtr, ytr, _, _ = make_split("and", 200, 0.2, random.Random(1))
        flipped = sum(y != TASKS["and"](x) for x, y in zip(Xtr, ytr)) / 200
        self.assertTrue(0.1 < flipped < 0.3)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = run(tasks=["and", "xor"], sizes=(64,), noises=(0.0,), reps=2)

    def test_every_learner_reports_a_valid_accuracy_on_every_task(self):
        self.assertEqual(len(self.records), 2 * len(LEARNERS) * 2)
        self.assertTrue(all(0.0 <= r["accuracy"] <= 1.0 for r in self.records))
        self.assertEqual({r["tribe"] for r in self.records}, set(TRIBES.values()))

    def test_runs_are_reproducible(self):
        again = run(tasks=["and", "xor"], sizes=(64,), noises=(0.0,), reps=2)
        self.assertEqual([r["accuracy"] for r in again], [r["accuracy"] for r in self.records])

    def test_expected_strengths_and_weaknesses(self):
        # a tree finds a two-feature rule exactly; naive Bayes cannot see an interaction
        self.assertGreaterEqual(mean_acc(self.records, task="and", learner="tree"), 0.99)
        self.assertLess(mean_acc(self.records, task="xor", learner="naive_bayes"), 0.6)
        self.assertGreater(mean_acc(self.records, task="xor", learner="mlp"), 0.9)

    def test_report_and_csv(self):
        text = report(self.records)
        self.assertIn("naive_bayes", text)
        self.assertIn("xor: x0 XOR x1", text)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.csv")
            write_csv(self.records, path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.records))

    def test_make_rejects_unknown_learner(self):
        with self.assertRaises(KeyError):
            make("oracle")


if __name__ == "__main__":
    unittest.main()
