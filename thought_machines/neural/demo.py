from .geometry import VECTORS, nearest, similarity_matrix
from .mlp import MLP
from .perceptron import GATES, Perceptron


def main():
    print("== One neuron ==")
    for name, data in GATES.items():
        p = Perceptron(2, seed=1)
        hist = p.fit(data, epochs=500)
        print(f"{name:>3}: accuracy {p.accuracy(data):.2f} after {len(hist)} epochs "
              f"(errors in last epoch: {hist[-1]})")
    print("XOR is not linearly separable: no line gets all 4 right, so training never settles.")

    print("\n== One hidden layer ==")
    data = GATES["XOR"]
    net = MLP(2, 4, seed=1)
    for epoch, loss in net.fit(data, epochs=3000):
        print(f"  epoch {epoch:>4}  loss {loss:.4f}")
    print("XOR accuracy:", net.accuracy(data), [round(net.predict_proba(x), 2) for x, _ in data])
    solved = 0
    for seed in range(10):
        n = MLP(2, 4, seed=seed)
        n.fit(data, epochs=3000, seed=seed)
        solved += n.accuracy(data) == 1.0
    print(f"solved from {solved} of 10 random starts (the rest sit on a flat region of the loss)")

    print("\n== Concepts as geometry (toy features) ==")
    for name in ("dog", "lion", "car"):
        print(f"  nearest to {name}:", ", ".join(f"{n} {s:.2f}" for n, s in nearest(name, 3)))
    sims = similarity_matrix()
    print(f"  cos(dog, wolf)={sims['dog']['wolf']:.2f}  cos(dog, truck)={sims['dog']['truck']:.2f}")
    assert set(VECTORS) == set(sims)


if __name__ == "__main__":
    main()
