import statistics as st

from .gridworld import CLIFF, SMALL_ROOM, GridWorld
from .td import (actor_critic, dopamine_conditioning, greedy_path, mc_prediction, q_learning, sarsa,
                 td0_prediction)


def main():
    print("== Prediction: TD(0) versus Monte Carlo on a five-state random walk ==")
    print("true values 1/6 .. 5/6; RMS error of the value estimates, mean of 100 runs, alpha 0.1")
    for n in (10, 100):
        td = st.mean(td0_prediction(n, 0.1, s)[1][-1] for s in range(100))
        mc = st.mean(mc_prediction(n, 0.1, s)[1][-1] for s in range(100))
        print(f"  after {n:>3} episodes: TD(0) {td:.3f}   Monte Carlo {mc:.3f}")
    print("TD learns from each step; Monte Carlo waits for the end of the episode.")

    print("\n== Control on the cliff: Q-learning versus SARSA ==")
    env = GridWorld(CLIFF)
    print(f"shortest path: {env.shortest_path_length()} steps, along the edge of the cliff")
    for name, learner in (("Q-learning", q_learning), ("SARSA", sarsa)):
        q, returns = learner(env, episodes=2000, seed=0)
        total, path, ok = greedy_path(env, q)
        print(f"  {name:<10} greedy path {len(path) - 1:>2} steps (reward {total:.0f}), "
              f"reward per episode while learning {st.mean(returns[-100:]):>6.1f}")
    print("Q-learning learns the shortest path and keeps falling off it while exploring.")
    print("SARSA learns about the policy it actually follows, so it takes the safer, longer path.")

    print("\n== Actor-critic: one TD error trains both a critic and an actor ==")
    room = GridWorld(SMALL_ROOM)
    prefs, v, returns = actor_critic(room, episodes=300, seed=0)
    total, path, ok = greedy_path(room, prefs)
    print(f"  first episode reward {returns[0]:.0f}, last {returns[-1]:.0f}, "
          f"greedy path {len(path) - 1} steps (optimal {room.shortest_path_length()})")

    print("\n== The dopamine story: where does the prediction error land? ==")
    history = dopamine_conditioning(trials=300)
    print(f"{'trial':>7} {'error at cue':>14} {'error at reward':>17}")
    for i in (0, 5, 20, 100, 299):
        print(f"{i + 1:>7} {history[i]['cue']:>14.3f} {history[i]['reward']:>17.3f}")
    print("Early on the reward is a surprise. After learning it is fully predicted, and the")
    print("surprise moves to the earliest cue that predicts it.")


if __name__ == "__main__":
    main()
