import unittest

from thought_machines.neural.geometry import ANIMALS, VEHICLES, VECTORS, cosine, nearest
from thought_machines.neural.mlp import MLP
from thought_machines.neural.perceptron import GATES, Perceptron


class PerceptronTests(unittest.TestCase):
    def test_and_or_are_learned(self):
        for name in ("AND", "OR"):
            p = Perceptron(2, seed=1)
            p.fit(GATES[name], epochs=500)
            self.assertEqual(p.accuracy(GATES[name]), 1.0, name)

    def test_xor_is_never_solved_by_one_neuron(self):
        for seed in range(10):
            p = Perceptron(2, seed=seed)
            p.fit(GATES["XOR"], epochs=300)
            self.assertLess(p.accuracy(GATES["XOR"]), 1.0)


class MLPTests(unittest.TestCase):
    def test_hidden_layer_solves_xor(self):
        solved = 0
        for seed in range(5):
            net = MLP(2, 4, seed=seed)
            net.fit(GATES["XOR"], epochs=3000, seed=seed)
            solved += net.accuracy(GATES["XOR"]) == 1.0
        self.assertGreaterEqual(solved, 4)

    def test_loss_goes_down(self):
        net = MLP(2, 4, seed=0)
        hist = net.fit(GATES["XOR"], epochs=1500)
        self.assertLess(hist[-1][1], hist[0][1])


class GeometryTests(unittest.TestCase):
    def test_categories_form_clusters(self):
        within = [cosine(VECTORS[a], VECTORS[b]) for a in ANIMALS for b in ANIMALS if a < b]
        across = [cosine(VECTORS[a], VECTORS[v]) for a in ANIMALS for v in VEHICLES]
        self.assertGreater(min(within), max(across))

    def test_truck_is_nearest_to_car(self):
        self.assertEqual(nearest("car", 1)[0][0], "truck")


if __name__ == "__main__":
    unittest.main()
