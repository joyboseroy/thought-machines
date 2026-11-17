from .kernel import KernelSVM, accuracy, linear, make_circles, rbf
from .knn import KNN


def main():
    Xtr, ytr = make_circles(120, noise=0.07, seed=1)
    Xte, yte = make_circles(200, noise=0.07, seed=2)
    print("Two concentric rings. No straight line separates them.")
    knn = KNN(k=3).fit(Xtr, ytr)
    print(f"3-NN accuracy:             {accuracy(knn, Xte, yte):.2f}")
    lin = KernelSVM(linear, epochs=20).fit(Xtr, ytr)
    print(f"SVM, linear kernel:        {accuracy(lin, Xte, yte):.2f}")
    r = KernelSVM(rbf(2.0), epochs=20).fit(Xtr, ytr)
    print(f"SVM, RBF kernel:           {accuracy(r, Xte, yte):.2f}  ({r.n_support()} support vectors of {len(Xtr)})")
    print("The RBF learner never builds a rule. It only compares new points to stored ones.")

    xor = [(0, 0), (0, 1), (1, 0), (1, 1)]
    y = [0, 1, 1, 0]
    print("\nXOR, linear vs RBF:",
          accuracy(KernelSVM(linear, epochs=50).fit(xor, y), xor, y),
          accuracy(KernelSVM(rbf(2.0), epochs=50).fit(xor, y), xor, y))


if __name__ == "__main__":
    main()
