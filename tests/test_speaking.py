import random
import statistics as st
import unittest

from thought_machines.speaking import NGramModel, ToyGrammar, cumulative_culture, noisy_channel_repair
from thought_machines.speaking.ngram import corrupt
from thought_machines.speaking.ratchet import lifetime_learning, transmit


class NGramTests(unittest.TestCase):
    def setUp(self):
        self.train = ToyGrammar(0).corpus(300)
        self.test = ToyGrammar(1).corpus(200)

    def test_more_context_lowers_perplexity(self):
        p = [NGramModel(n, 0.1).fit(self.train).perplexity(self.test) for n in (1, 2, 3)]
        self.assertGreater(p[0], 2 * p[1])
        self.assertGreater(p[1], p[2])

    def test_unsmoothed_model_breaks_on_unseen_context(self):
        self.assertEqual(NGramModel(3, 0.0).fit(self.train[:30]).perplexity(self.test), float("inf"))

    def test_smoothing_gives_finite_perplexity(self):
        self.assertLess(NGramModel(3, 0.1).fit(self.train[:30]).perplexity(self.test), 10)

    def test_probabilities_sum_to_one(self):
        m = NGramModel(2, 0.1).fit(self.train)
        for ctx in ([], ["the"], ["the", "dog"]):
            self.assertAlmostEqual(sum(m.prob(ctx, w) for w in m.vocab), 1.0)

    def test_generated_sentences_respect_agreement_mostly(self):
        m = NGramModel(3, 0.01).fit(self.train)
        singular = {"dog", "cat", "bird"}
        good = bad = 0
        for seed in range(60):
            s = m.generate(seed)
            if len(s) >= 3 and s[0] in ("the", "a") and s[1] in singular | {"dogs", "cats"}:
                plural = s[1].endswith("s")
                verb_ok = s[2] in ({"see", "chase", "like"} if plural else {"sees", "chases", "likes"})
                good += verb_ok
                bad += not verb_ok
        self.assertGreater(good, 5 * max(bad, 1) - 1)

    def test_bad_order_rejected(self):
        with self.assertRaises(ValueError):
            NGramModel(0)

    def test_repair_improves_with_context(self):
        results = []
        for n in (1, 3):
            model = NGramModel(n, 0.01).fit(self.train)
            vocab = sorted(model.vocab - {"</s>"})
            rng, ok = random.Random(0), 0
            for s in self.test[:100]:
                pos = rng.randrange(len(s))
                wrong = rng.choice([w for w in vocab if w != s[pos]])
                ok += noisy_channel_repair(model, corrupt(s, pos, wrong), pos) == s
            results.append(ok)
        self.assertGreater(results[1], results[0] + 30)
        self.assertGreater(results[1], 50)


class RatchetTests(unittest.TestCase):
    def mean_final(self, **kw):
        return st.mean(cumulative_culture(seed=s, **kw)[-1] for s in range(6))

    def test_without_language_there_is_no_progress(self):
        self.assertLess(self.mean_final(social=False), 0.8)

    def test_with_language_knowledge_accumulates(self):
        self.assertGreater(self.mean_final(social=True, error_rate=0.0), 0.97)

    def test_noisy_channel_caps_the_ratchet(self):
        clean = self.mean_final(social=True, error_rate=0.0)
        noisy = self.mean_final(social=True, error_rate=0.4)
        self.assertGreater(clean - noisy, 0.15)

    def test_more_noise_never_helps(self):
        levels = [self.mean_final(social=True, error_rate=e) for e in (0.0, 0.1, 0.3)]
        self.assertGreaterEqual(levels[0], levels[1])
        self.assertGreaterEqual(levels[1], levels[2])

    def test_lifetime_learning_never_gets_worse(self):
        rng = random.Random(0)
        target = [rng.randrange(2) for _ in range(40)]
        start = [rng.randrange(2) for _ in range(40)]
        before = sum(a == b for a, b in zip(start, target))
        _, after = lifetime_learning(start, target, 100, rng)
        self.assertGreaterEqual(after, before)

    def test_transmit_extremes(self):
        rng = random.Random(0)
        msg = [0, 1] * 20
        self.assertEqual(transmit(msg, 0.0, rng), msg)
        self.assertEqual(transmit(msg, 1.0, rng), [1 - b for b in msg])

    def test_reproducible(self):
        self.assertEqual(cumulative_culture(seed=3), cumulative_culture(seed=3))


if __name__ == "__main__":
    unittest.main()
