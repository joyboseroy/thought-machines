"""Simulating: learn a model of the world, then think with it.

Bennett's third breakthrough, in early mammals, is simulating: rehearsing actions in an
internal model before (or instead of) trying them. The classic algorithm is Dyna-Q.
Every real step does two things. It updates action values from the real experience,
and it records the transition in a model. Then, between real steps, it replays
transitions drawn from the model and updates the same values, as if the experience had
happened again. Thinking is cheap and the world is not.

Two results here:

* Planning steps convert a little experience into a lot of learning: more simulated
  updates per real step, fewer real steps to find the goal.
* Latent learning (Tolman): explore a maze with no reward, learn nothing about reward,
  and still be near-optimal almost at once when a reward finally appears, because the
  map was already in the model. A model-free learner has to start from zero.

`plan_by_search` adds the bridge to the `search` pillar: once there is a model, finding
a route is just search over it, with no further contact with the world.
"""

from __future__ import annotations

import random

from ..reinforcement.gridworld import GridWorld
from ..search.core import search


class DynaQ:
    """Tabular Dyna-Q on a deterministic grid world. `n_planning = 0` is plain Q-learning."""

    def __init__(self, env: GridWorld, n_planning: int = 0, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, seed: int = 0):
        self.env, self.n_planning = env, n_planning
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = random.Random(seed)
        self.q = {s: [0.0] * 4 for s in env.states()}
        self.model = {}       # (state, action) -> (reward, next_state, done)
        self._seen = []       # the keys of the model, for uniform sampling
        self.real_steps = 0

    def choose(self, state, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return self.rng.randrange(4)
        best = max(self.q[state])
        return self.rng.choice([a for a in range(4) if self.q[state][a] == best])

    def _update(self, s, a, r, nxt, done):
        target = r + (0.0 if done else self.gamma * max(self.q[nxt]))
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    def step(self, state, epsilon=None):
        a = self.choose(state, epsilon)
        nxt, r, done = self.env.step(state, a)
        self._update(state, a, r, nxt, done)
        if (state, a) not in self.model:
            self._seen.append((state, a))
        self.model[(state, a)] = (r, nxt, done)
        for _ in range(self.n_planning):
            ps, pa = self._seen[self.rng.randrange(len(self._seen))]
            pr, pn, pd = self.model[(ps, pa)]
            self._update(ps, pa, pr, pn, pd)
        self.real_steps += 1
        return nxt, done

    def run_episode(self, max_steps: int = 5000):
        state, steps, done = self.env.start, 0, False
        while not done and steps < max_steps:
            state, done = self.step(state)
            steps += 1
        return steps


def learning_curve(env: GridWorld, n_planning: int, episodes: int = 30, seed: int = 0):
    agent = DynaQ(env, n_planning=n_planning, seed=seed)
    return [agent.run_episode() for _ in range(episodes)]


def latent_learning(env: GridWorld, n_planning: int, explore_steps: int = 3000, episodes: int = 6,
                    goal_reward: float = 1.0, seed: int = 0):
    """Explore with no reward at all, then switch the reward on. Returns steps per episode.

    The environment is restored to its original reward settings afterwards.
    """
    saved = (env.step_reward, env.goal_reward)
    env.step_reward, env.goal_reward = 0.0, 0.0
    agent = DynaQ(env, n_planning=n_planning, seed=seed)
    state = env.start
    for _ in range(explore_steps):  # a random walk that keeps restarting at the goal
        state, done = agent.step(state, epsilon=1.0)
        if done:
            state = env.start
    env.goal_reward = goal_reward
    try:
        return [agent.run_episode() for _ in range(episodes)]
    finally:
        env.step_reward, env.goal_reward = saved


class LearnedModelProblem:
    """The learned model, dressed as a search problem. No contact with the real world."""

    def __init__(self, agent: DynaQ):
        self.initial = agent.env.start
        self._edges = {}
        self._goals = set()
        for (s, a), (_, nxt, done) in agent.model.items():
            self._edges.setdefault(s, []).append((a, nxt))
            if done:
                self._goals.add(nxt)

    def is_goal(self, state):
        return state in self._goals

    def successors(self, state):
        for a, nxt in self._edges.get(state, []):
            yield a, nxt, 1.0


def plan_by_search(agent: DynaQ, strategy: str = "bfs"):
    """Find a route using only the learned model. Returns a list of actions, or None."""
    result = search(LearnedModelProblem(agent), strategy)
    return result.path if result.found else None
