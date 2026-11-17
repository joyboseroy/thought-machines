from .ga import genetic_algorithm
from .gp import evolve, regression_language, to_str


def main():
    print("== Genetic algorithm: OneMax on 40 bits ==")
    best, f, hist = genetic_algorithm(lambda b: sum(b), 40, seed=1)
    print(f"best fitness {f} of 40; reached {hist[10]} after 10 generations")

    print("\n== Genetic programming: discover x^2 + 3x + 1 from samples ==")
    xs = list(range(-5, 6))
    target = lambda x: x * x + 3 * x + 1
    lang = regression_language()

    def fitness(tree):
        from .gp import evaluate
        return sum((evaluate(tree, lang, {"x": x}) - target(x)) ** 2 for x in xs) / len(xs)

    for seed in range(3):
        r = evolve(lang, fitness, pop_size=300, generations=60, parsimony=0.001, seed=seed)
        status = "exact" if r.error < 1e-9 else f"mse {r.error:.3f}"
        print(f"seed {seed}: {to_str(r.tree)}   ({status}, {r.generations} generations)")
    print("Gradient descent would need the form chosen for it. Evolution searched the forms.")


if __name__ == "__main__":
    main()
