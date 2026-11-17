from .inference import posterior, sequential_update
from .mcmc import coin_log_posterior, metropolis
from .naive_bayes import NaiveBayes
from .network import burglary_alarm
from .number_game import PRIORS, generalise, infer, prior_over, top


def main():
    print("== Medical test ==")
    p = posterior(0.01, 0.95, 0.05)
    print(f"prevalence 1%, sensitivity 95%, false positive 5%: P(disease | positive) = {p:.3f}")
    path = sequential_update(0.01, 0.95, 0.05, [True, True, True, False])
    print("belief after +,+,+,-:", " -> ".join(f"{b:.3f}" for b in path))

    print("\n== Number game ==")
    prior = prior_over(PRIORS["balanced"])
    for data in ([16], [16, 8, 2, 64], [16, 23, 19, 20]):
        post = infer(data, prior)
        g = generalise(post)
        best = ", ".join(f"{n} ({p:.2f})" for n, p in top(post, 3))
        print(f"data {data}: top hypotheses {best}")
        print(f"   P(32 in concept)={g[32]:.2f}  P(17)={g[17]:.2f}  P(30)={g[30]:.2f}  P(99)={g[99]:.2f}")

    print("\nSame data [16], different inductive bias:")
    for name in ("mathy-only", "interval-only", "balanced"):
        g = generalise(infer([16], prior_over(PRIORS[name])))
        print(f"  {name:>13}: P(17)={g[17]:.2f}  P(32)={g[32]:.2f}  P(18)={g[18]:.2f}")

    print("\n== Bayesian network (burglary and alarm) ==")
    net = burglary_alarm()
    print(f"P(burglary | John and Mary call) = {net.prob({'Burglary': True}, {'JohnCalls': True, 'MaryCalls': True}):.3f}")
    print(f"P(burglary | alarm, earthquake)  = {net.prob({'Burglary': True}, {'Alarm': True, 'Earthquake': True}):.3f}  (explaining away)")

    print("\n== MCMC: bias of a coin after 7 heads, 3 tails ==")
    samples, acc = metropolis(coin_log_posterior(7, 3), 0.5, step=0.2)
    print(f"sample mean {sum(samples) / len(samples):.3f}  exact Beta(8,4) mean {8 / 12:.3f}  acceptance {acc:.2f}")

    print("\n== Naive Bayes vs interaction ==")
    X = [(a, b) for a in (0, 1) for b in (0, 1)] * 10
    for name, y in (("AND", [a & b for a, b in X]), ("XOR", [a ^ b for a, b in X])):
        nb = NaiveBayes().fit(X, y)
        acc = sum(nb.predict(x) == t for x, t in zip(X, y)) / len(X)
        print(f"  {name}: accuracy {acc:.2f}")
    print("Independence assumptions cannot see a pure interaction.")


if __name__ == "__main__":
    main()
