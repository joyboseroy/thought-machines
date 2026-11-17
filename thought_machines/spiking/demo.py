from .izhikevich import PRESETS, cv, simulate_izhikevich
from .lif import analytic_rate_hz, rate_hz, simulate_lif
from .rank_order import distortion_experiment
from .stdp import simulate_stdp, stdp_window


def main():
    print("== Leaky integrate-and-fire: firing rate versus input ==")
    print(f"{'input':>6} {'simulated Hz':>13} {'analytic Hz':>12}")
    for i in (10, 16, 20, 30, 50):
        spikes, _ = simulate_lif(i, t_max=2000)
        print(f"{i:>6} {rate_hz(spikes, 2000):>13.1f} {analytic_rate_hz(i):>12.1f}")
    print("Below an input of 15 the neuron never reaches threshold.")

    print("\n== Izhikevich neuron, constant input of 10, one second ==")
    for name, (a, b, c, d) in PRESETS.items():
        s = simulate_izhikevich(a, b, c, d, 10.0)
        print(f"  {name:<22} {len(s):>3} spikes, ISI variability (CV) {cv(s):.2f}")

    print("\n== STDP ==")
    print("weight change for pre-before-post (+5 ms):", f"{stdp_window(5):+.4f}",
          "  post-before-pre (-5 ms):", f"{stdp_window(-5):+.4f}")
    w = simulate_stdp(seconds=60, seed=1)
    corr, unc = w[:20], w[20:]
    print(f"after 60 s: mean weight of the 20 correlated inputs {sum(corr) / 20:.2f}, "
          f"of the 20 uncorrelated inputs {sum(unc) / 20:.2f} (started at 0.50)")
    print("Both groups fire at the same average rate, so only spike timing separates them.")

    print("\n== Rank-order code ==")
    print("8 similar templates, 30 inputs, each trial distorted by x -> x**gamma with random gamma:")
    for k, v in distortion_experiment(trials=400, seed=0).items():
        print(f"  {k:<11} {v:.2f}")
    print("The firing order is untouched by any monotonic distortion. Cosine copes with gain but not with")
    print("curvature, and Euclidean distance with neither. The margin is modest on this easy task.")


if __name__ == "__main__":
    main()
