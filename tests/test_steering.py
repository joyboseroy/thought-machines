import math
import statistics as st
import unittest

from thought_machines.steering import (RunAndTumble, braitenberg_run, chemotaxis_trial,
                                       concentration, random_walk_trial)


class SteeringTests(unittest.TestCase):
    def test_field_peaks_at_the_source_and_falls_with_distance(self):
        self.assertEqual(concentration(0, 0), 1.0)
        self.assertGreater(concentration(1, 0), concentration(2, 0))

    def test_tumble_rate_depends_on_whether_things_improve(self):
        up = RunAndTumble(p_better=0.02, p_worse=0.5, valence=1)
        down = RunAndTumble(p_better=0.02, p_worse=0.5, valence=-1)
        self.assertEqual(up.tumble_probability(+0.1), 0.02)
        self.assertEqual(up.tumble_probability(-0.1), 0.5)
        self.assertEqual(down.tumble_probability(+0.1), 0.5)
        self.assertEqual(down.tumble_probability(-0.1), 0.02)

    def test_bad_valence_rejected(self):
        with self.assertRaises(ValueError):
            RunAndTumble(valence=0)

    def test_run_and_tumble_beats_a_blind_walk_by_a_wide_margin(self):
        seeds = range(40)
        rt = st.mean(chemotaxis_trial(RunAndTumble(seed=s))[-1] for s in seeds)
        blind = st.mean(random_walk_trial(s, rate=0.1)[-1] for s in seeds)
        self.assertLess(rt, 15)
        self.assertGreater(blind, 3 * rt)

    def test_flipping_the_sign_of_good_makes_the_agent_flee(self):
        seeds = range(40)
        approach = st.mean(chemotaxis_trial(RunAndTumble(seed=s), start=(5.0, 0.0))[-1] for s in seeds)
        avoid = st.mean(chemotaxis_trial(RunAndTumble(valence=-1, seed=s), start=(5.0, 0.0))[-1]
                        for s in seeds)
        self.assertLess(approach, 12.0)  # hovers near the source, overshooting by a few steps
        self.assertGreater(avoid, 50.0)
        self.assertGreater(avoid, 5 * approach)

    def test_trials_are_reproducible(self):
        a = chemotaxis_trial(RunAndTumble(seed=7))
        b = chemotaxis_trial(RunAndTumble(seed=7))
        self.assertEqual(a, b)

    def test_crossed_vehicle_arrives_and_straight_vehicle_flees(self):
        for heading in (math.pi / 2, 2.0, 2.8):
            crossed = braitenberg_run(True, heading=heading)
            straight = braitenberg_run(False, heading=heading)
            self.assertLessEqual(math.hypot(*crossed[-1]), 1.5)
            self.assertGreater(math.hypot(*straight[-1]), 40.0)

    def test_the_vehicle_is_not_magic_when_facing_away(self):
        # Pointing away from the source, both sensors read almost the same, so
        # crossed wiring has nothing to turn toward. It does not arrive.
        self.assertGreater(math.hypot(*braitenberg_run(True, heading=0.3)[-1]), 20.0)


if __name__ == "__main__":
    unittest.main()
