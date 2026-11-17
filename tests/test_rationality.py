import contextlib
import io
import unittest

from thought_machines.cli import PILLARS, main
from thought_machines.rationality.bounded import best_row, compare


class RationalityTests(unittest.TestCase):
    def test_free_thinking_favours_the_ideal_agent(self):
        rows = compare(n_options=256, trials=600, cost=0.0)
        self.assertTrue(best_row(rows).name.startswith("ideal"))

    def test_costly_thinking_favours_a_bounded_agent(self):
        rows = compare(n_options=256, trials=600, cost=0.01)
        ideal = next(r for r in rows if r.name.startswith("ideal"))
        best = best_row(rows)
        self.assertLess(best.looks, 256)
        self.assertGreater(best.net, ideal.net)
        self.assertGreater(best.regret, ideal.regret)  # worse choices, better outcome

    def test_optimal_budget_shrinks_as_cost_rises(self):
        ks = [best_row(compare(n_options=256, trials=600, cost=c)).looks
              for c in (0.0, 0.001, 0.01, 0.1)]
        self.assertEqual(ks, sorted(ks, reverse=True))

    def test_reproducible(self):
        a = compare(n_options=64, trials=100, cost=0.02, seed=7)
        b = compare(n_options=64, trials=100, cost=0.02, seed=7)
        self.assertEqual(a, b)


class DemoSmokeTests(unittest.TestCase):
    def test_every_demo_runs(self):
        for name in PILLARS:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(["demo", name]), 0)
            self.assertGreater(len(buf.getvalue()), 50, name)


if __name__ == "__main__":
    unittest.main()
