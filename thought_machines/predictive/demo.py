from .coding import grid_map, infer_by_gradient, thermostat


def main():
    u, mean, pvar, nvar = 2.0, 3.0, 1.0, 1.0
    print("== Perception as inference ==")
    print(f"prior belief about the cause: {mean}; observation: {u}; model: u = v^2 + noise")
    path = infer_by_gradient(u, mean, pvar, nvar)
    print(f"gradient descent on prediction errors settles at v = {path[-1]:.3f}")
    print(f"grid search for the posterior mode gives      v = {grid_map(u, mean, pvar, nvar):.3f}")
    print("belief trajectory:", " ".join(f"{path[i]:.2f}" for i in (0, 25, 50, 100, 200, 400, 1000)))

    print("\n== Acting to fulfil a prediction (toy active inference) ==")
    t = thermostat()
    print("temperature:", " ".join(f"{t[i]:.1f}" for i in (0, 5, 10, 20, 40, 60)))
    print("The agent expects 22. Rather than update the belief, it changes the world.")


if __name__ == "__main__":
    main()
