import random
import unittest

from thought_machines.active.design import (entropy, expected_info_gain, run_session,
                                            threshold_world)
from thought_machines.causal.intervene import (average_effect, backdoor, confounded_net,
                                               confounded_net_with_effect, intervene, observe,
                                               rain_net)
from thought_machines.memory.hopfield import Hopfield, corrupt, random_pattern, recall_rate
from thought_machines.memory.sdm import SDM
from thought_machines.predictive.coding import grid_map, infer_by_gradient, thermostat
from thought_machines.spiking.izhikevich import PRESETS, cv, simulate_izhikevich
from thought_machines.spiking.lif import analytic_rate_hz, rate_hz, simulate_lif
from thought_machines.spiking.rank_order import (RankOrderClassifier, distortion_experiment,
                                                 firing_order, make_task, similarity)
from thought_machines.spiking.stdp import simulate_stdp, stdp_window


class CausalTests(unittest.TestCase):
    def test_confounding_makes_a_cause_out_of_nothing(self):
        net = confounded_net()
        self.assertAlmostEqual(observe(net, "Drowning", "IceCream"), 0.25)
        self.assertAlmostEqual(intervene(net, "Drowning", "IceCream"), net.prob({"Drowning": True}))
        self.assertAlmostEqual(average_effect(net, "Drowning", "IceCream"), 0.0)

    def test_backdoor_adjustment_equals_the_intervention(self):
        for net in (confounded_net(), confounded_net_with_effect()):
            for value in (True, False):
                self.assertAlmostEqual(backdoor(net, "Drowning", "IceCream", ["Hot"], value),
                                       intervene(net, "Drowning", "IceCream", value))

    def test_observed_difference_overstates_the_true_effect(self):
        net = confounded_net_with_effect()
        seen = observe(net, "Drowning", "IceCream") - observe(net, "Drowning", "IceCream", False)
        self.assertAlmostEqual(average_effect(net, "Drowning", "IceCream"), 0.04)
        self.assertGreater(seen, 4 * 0.04)

    def test_forcing_an_effect_does_not_change_its_cause(self):
        rain = rain_net()
        self.assertAlmostEqual(intervene(rain, "WetGrass", "Umbrella"), rain.prob({"WetGrass": True}))
        self.assertGreater(observe(rain, "WetGrass", "Umbrella"), intervene(rain, "WetGrass", "Umbrella"))
        self.assertGreater(intervene(rain, "WetGrass", "Rain"), rain.prob({"WetGrass": True}))


class ActiveTests(unittest.TestCase):
    def test_information_gain_is_one_bit_for_a_fair_split_and_never_exceeds_entropy(self):
        prior, p1, queries = threshold_world(64)
        self.assertAlmostEqual(expected_info_gain(prior, p1, 31), 1.0)
        self.assertLess(expected_info_gain(prior, p1, 3), 1.0)
        self.assertTrue(all(expected_info_gain(prior, p1, q) <= entropy(prior) + 1e-9 for q in queries))

    def test_eig_rediscovers_binary_search(self):
        prior, p1, queries = threshold_world(64)
        asked, belief = run_session(prior, p1, queries, lambda q: int(q >= 37))
        self.assertEqual(len(asked), 6)  # log2(64)
        self.assertEqual(asked[0], 31)  # the median
        self.assertEqual(max(belief, key=belief.get), 37)

    def test_random_questions_do_not_get_there(self):
        prior, p1, queries = threshold_world(64)
        asked, belief = run_session(prior, p1, queries, lambda q: int(q >= 37),
                                    chooser="random", max_queries=20)
        self.assertEqual(len(asked), 20)
        self.assertLess(max(belief.values()), 0.99)

    def test_noisy_answers_cost_more_questions_but_still_converge(self):
        prior, p1, queries = threshold_world(32, noise=0.1)
        noise = random.Random(5)

        def answer(q):
            truth = int(q >= 20)
            return truth if noise.random() > 0.1 else 1 - truth

        asked, belief = run_session(prior, p1, queries, answer, max_queries=60)
        self.assertGreater(len(asked), 5)
        self.assertEqual(max(belief, key=belief.get), 20)


class PredictiveTests(unittest.TestCase):
    def test_gradient_on_prediction_errors_finds_the_exact_posterior_mode(self):
        path = infer_by_gradient(2.0, 3.0, 1.0, 0.5)
        self.assertAlmostEqual(path[-1], grid_map(2.0, 3.0, 1.0, 0.5), places=2)

    def test_belief_lands_between_prior_and_the_data(self):
        v = infer_by_gradient(2.0, 3.0, 1.0, 0.5)[-1]
        self.assertLess(v, 3.0)
        self.assertGreater(v, 2.0 ** 0.5)  # the value the data alone would suggest

    def test_acting_pulls_the_world_to_the_expectation(self):
        path = thermostat()
        self.assertLess(abs(path[-1] - 21.75), 0.05)  # target minus leak/gain
        self.assertEqual(path, sorted(path))


class MemoryTests(unittest.TestCase):
    def test_hopfield_recalls_when_lightly_loaded_and_fails_when_overloaded(self):
        self.assertGreaterEqual(recall_rate(100, 3), 0.95)
        self.assertLess(recall_rate(100, 30), 0.2)

    def test_recall_never_raises_energy(self):
        rng = random.Random(1)
        pats = [random_pattern(60, rng) for _ in range(3)]
        net = Hopfield(60).store(pats)
        cue = corrupt(pats[0], 0.2, rng)
        self.assertLessEqual(net.energy(net.recall(cue)), net.energy(cue))

    def test_sdm_cleans_up_a_noisy_address(self):
        sdm = SDM()
        words = [sdm.random_word() for _ in range(10)]
        for w in words:
            sdm.write(w, w)
        hits = sum(sdm.recall(sdm.noisy(w, 10)) == w for w in words)
        self.assertGreaterEqual(hits, 8)

    def test_sdm_has_a_critical_distance_in_a_loaded_memory(self):
        # a nearly empty memory forgives anything; the cliff appears once words compete
        for seed in (0, 1, 2):
            sdm = SDM(seed=seed)
            words = [sdm.random_word() for _ in range(10)]
            for w in words:
                sdm.write(w, w)
            near = sum(sdm.recall(sdm.noisy(w, 10)) == w for w in words)
            far = sum(sdm.recall(sdm.noisy(w, 100)) == w for w in words)
            self.assertGreaterEqual(near, 8)
            self.assertLessEqual(far, 6)


class SpikingTests(unittest.TestCase):
    def test_lif_matches_the_analytic_firing_rate(self):
        for current in (16, 20, 30):
            spikes, _ = simulate_lif(current, t_max=2000)
            self.assertAlmostEqual(rate_hz(spikes, 2000), analytic_rate_hz(current), delta=1.5)

    def test_lif_is_silent_below_threshold_and_faster_with_more_drive(self):
        self.assertEqual(simulate_lif(10, t_max=1000)[0], [])
        rates = [rate_hz(simulate_lif(i, t_max=1000)[0], 1000) for i in (16, 20, 30)]
        self.assertEqual(rates, sorted(rates))

    def test_izhikevich_presets_have_distinct_firing_statistics(self):
        stats = {name: simulate_izhikevich(*p, 10.0) for name, p in PRESETS.items()}
        self.assertTrue(all(len(s) > 5 for s in stats.values()))
        self.assertGreater(cv(stats["chattering"]), 3 * cv(stats["regular spiking"]))
        self.assertGreater(len(stats["fast spiking"]), len(stats["regular spiking"]))

    def test_stdp_window_has_the_right_shape(self):
        self.assertGreater(stdp_window(5), 0)
        self.assertLess(stdp_window(-5), 0)
        self.assertGreater(stdp_window(2), stdp_window(20))
        self.assertGreater(abs(stdp_window(-5)), stdp_window(5))  # depression slightly wins

    def test_stdp_selects_the_correlated_inputs_although_rates_are_equal(self):
        for seed in (0, 1):
            w = simulate_stdp(seconds=60, seed=seed)
            corr, unc = sum(w[:20]) / 20, sum(w[20:]) / 20
            self.assertGreater(corr, 0.9)
            self.assertLess(unc, 0.6)

    def test_rank_order_ignores_monotonic_distortion(self):
        x = [0.9, 0.1, 0.5, 0.7]
        self.assertEqual(firing_order(x), firing_order([v ** 3 for v in x]))
        order = firing_order(x)
        self.assertAlmostEqual(similarity(order, order), 1.0)
        one_swap = [order[1], order[0]] + order[2:]
        # the further the firing order is from the template, the lower the match
        self.assertLess(similarity(order, order[::-1]), similarity(order, one_swap))
        self.assertLess(similarity(order, one_swap), similarity(order, order))

    def test_rank_order_beats_euclidean_under_distortion(self):
        r = distortion_experiment(trials=300)
        self.assertGreaterEqual(r["rank order"], 0.99)
        self.assertGreater(r["rank order"], r["euclidean"])
        clf = RankOrderClassifier().fit(make_task())
        self.assertIn(clf.predict(make_task()[3]), range(8))


if __name__ == "__main__":
    unittest.main()
