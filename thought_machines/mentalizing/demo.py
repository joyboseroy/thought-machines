import statistics as st

from .false_belief import sally_anne
from .inverse_planning import Room, goal_posterior, posterior_trace, sample_trajectory

LAYOUT = ["A.......B",
          ".........",
          "....S....",
          ".........",
          "....C...."]


def main():
    room = Room(LAYOUT)
    print("== Reading a goal from behaviour: Bayesian inverse planning ==")
    print("An agent starts at S and heads for one of three goals A, B, C. We watch it move.")
    path = sample_trajectory(room, "A", beta=3.0, steps=6, seed=1)
    print(f"{'steps seen':>10} {'P(A)':>7} {'P(B)':>7} {'P(C)':>7}")
    for t, post in enumerate(posterior_trace(room, path, beta=3.0)):
        print(f"{t:>10} {post['A']:>7.2f} {post['B']:>7.2f} {post['C']:>7.2f}")
    print("A and B are equally far at first, so the first move cannot separate them. The")
    print("second move can. Watching a few steps is enough to name the goal.")

    print("\n== How often is the guess right? ==")
    print("steps watched -> accuracy over 180 simulated agents, noisy agent (beta = 2)")
    for steps in (1, 2, 4, 6):
        hits = []
        for goal in "ABC":
            for seed in range(60):
                traj = sample_trajectory(room, goal, 2.0, steps, seed)
                post = goal_posterior(room, traj, 2.0)
                hits.append(max(post, key=post.get) == goal)
        print(f"  {steps} steps: {st.mean(hits):.2f}")
    print("A random agent (beta = 0) reveals nothing:")
    traj = sample_trajectory(room, "A", 0.0, 6, seed=1)
    post = goal_posterior(room, traj, 0.0)
    print("  posterior after 6 steps:", {k: round(v, 2) for k, v in post.items()})

    print("\n== False belief: the Sally-Anne task ==")
    for watches in (False, True):
        scene = sally_anne(sally_watches_the_move=watches)
        label = "Sally sees Anne move the marble" if watches else "Sally is out of the room when the marble moves"
        print(f"  {label}")
        print(f"    marble really is in the {scene.reality('marble')}")
        print(f"    predictor with no model of minds says Sally looks in the "
              f"{scene.predict_search('Sally', 'marble', mentalizing=False)}")
        print(f"    predictor that tracks who saw what says Sally looks in the "
              f"{scene.predict_search('Sally', 'marble', mentalizing=True)}")
    scene = sally_anne()
    print(f"  Anne's model of Sally: Sally thinks the marble is in the "
          f"{scene.belief_about_belief('Anne', 'Sally', 'marble')}")
    print("The two predictors agree whenever beliefs are true. They split exactly when someone is wrong.")


if __name__ == "__main__":
    main()
