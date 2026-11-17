import math
import statistics as st

from .taxis import RunAndTumble, braitenberg_run, chemotaxis_trial, random_walk_trial


def main():
    print("== Run and tumble: climb a gradient with a one-number memory ==")
    print("start 20 units from the source, 400 steps, mean over 60 seeds")
    seeds = range(60)
    runs = {
        "run-and-tumble (approach)": [chemotaxis_trial(RunAndTumble(seed=s)) for s in seeds],
        "random walk, same speed": [random_walk_trial(s, rate=0.1) for s in seeds],
        "run-and-tumble (avoid)": [chemotaxis_trial(RunAndTumble(valence=-1, seed=s)) for s in seeds],
    }
    for name, trials in runs.items():
        print(f"  {name:<28} final distance {st.mean(t[-1] for t in trials):>6.1f}   "
              f"closest approach {st.mean(min(t) for t in trials):>5.1f}")
    print("The only difference between approach and avoid is the sign of 'better'.")

    print("\n== Braitenberg vehicles: wiring, not representation ==")
    for crossed, label in ((True, "crossed wiring (turns toward)"), (False, "same-side wiring (turns away)")):
        path = braitenberg_run(crossed)
        d = [math.hypot(*p) for p in path]
        outcome = "arrived" if d[-1] <= 1.5 else "fled"
        print(f"  {label:<30} start 10.0, final {d[-1]:>5.1f}, {len(path) - 1:>3} steps ({outcome})")
    print("Neither vehicle has any idea where the source is. Valence is in the wiring.")


if __name__ == "__main__":
    main()
