from ..bayes.number_game import HYPOTHESES, PRIORS, prior_over
from .design import run_session, threshold_world


def main():
    print("== Find a hidden threshold among 64 values with yes/no questions ==")
    prior, p1, queries = threshold_world(64)
    truth = 37
    for chooser in ("eig", "random"):
        asked, belief = run_session(prior, p1, queries, lambda q: int(truth <= q), chooser, seed=3)
        done = max(belief.values()) >= 0.99
        print(f"{chooser:>6}: {len(asked):>2} questions, "
              f"{'found ' + str(max(belief, key=belief.get)) if done else 'still undecided at the cap'}")
    asked, _ = run_session(prior, p1, queries, lambda q: int(truth <= q), "eig")
    print("questions chosen by information gain:", asked)

    print("\n== Twenty questions on the number game (is x in the concept?) ==")
    hidden = next(h for h in HYPOTHESES if h.name == "powers of 2")
    prior = prior_over(PRIORS["balanced"])
    nums = list(range(1, 101))
    p1 = lambda h, q: 1.0 if q in h.members else 0.0
    asked, belief = run_session(prior, p1, nums, lambda q: int(q in hidden.members), max_queries=30)
    print(f"hidden concept: {hidden.name}")
    print(f"identified after {len(asked)} questions, asked about: {asked}")
    print("posterior winner:", max(belief, key=belief.get).name)


if __name__ == "__main__":
    main()
