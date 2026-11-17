import statistics as st

from ..reinforcement.gridworld import DYNA_MAZE, GridWorld
from .dyna import DynaQ, latent_learning, learning_curve, plan_by_search


def main():
    env = GridWorld(DYNA_MAZE, step_reward=0.0, goal_reward=1.0)
    best = env.shortest_path_length()
    print(f"A maze with two walls to get around. The shortest path is {best} steps.")

    print("\n== More simulated steps per real step means fewer real steps ==")
    print("steps to reach the goal per episode, mean of 10 runs (n = planning steps per real step)")
    print(f"{'n':>4} {'episode 1':>10} {'episode 2':>10} {'episode 10':>11} {'last 5':>8}")
    for n in (0, 5, 50):
        curves = [learning_curve(env, n, episodes=30, seed=s) for s in range(10)]
        mean = lambda i: st.mean(c[i] for c in curves)  # noqa: E731
        last = st.mean(st.mean(c[-5:]) for c in curves)
        print(f"{n:>4} {mean(0):>10.0f} {mean(1):>10.0f} {mean(9):>11.0f} {last:>8.1f}")
    print("n = 0 is plain Q-learning. The first episode is slow for everyone: nobody knows the maze yet.")

    print("\n== Latent learning: explore with no reward, then switch the reward on ==")
    print("3000 random steps with no reward anywhere, then steps per episode once the goal pays")
    for n in (0, 50):
        runs = [latent_learning(env, n, explore_steps=3000, episodes=6, seed=s) for s in range(10)]
        row = " ".join(f"{st.mean(r[i] for r in runs):>6.0f}" for i in range(6))
        label = "model-free (n = 0) " if n == 0 else "with a model (n = 50)"
        print(f"  {label:<22} episodes 1 to 6: {row}")
    print("The agent with a model has already learned the maze. One taste of reward and it replans.")

    print("\n== A model turns navigation into search ==")
    agent = DynaQ(env, n_planning=50, seed=0)
    for _ in range(10):
        agent.run_episode()
    path = plan_by_search(agent, "bfs")
    names = "URDL"
    print("after 10 episodes, breadth-first search over the learned model, no real steps:")
    print(f"  {len(path)} actions: {''.join(names[a] for a in path)}  (shortest possible: {best})")


if __name__ == "__main__":
    main()
