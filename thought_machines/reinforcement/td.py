"""Reinforcing: learn from consequences by temporal-difference error.

Bennett's second breakthrough, in early vertebrates, is learning from trial and error.
The computational core is the temporal-difference (TD) error

    delta = r + gamma * V(next) - V(now)

the gap between what happened and what was predicted. One number, used to nudge a
prediction (critic) and a choice (actor). The dopamine story in neuroscience is that
phasic dopamine signals this error, and `dopamine_conditioning` shows the signature
result: with training, the error moves from the reward to the cue that predicts it.

Contents: TD(0) and Monte Carlo prediction on a random walk, Q-learning and SARSA on
the cliff, a one-step actor-critic, and the conditioning demo.
"""

from __future__ import annotations

import math
import random

from .gridworld import GridWorld


# ----------------------------------------------------------------- prediction

class RandomWalk:
    """Sutton and Barto's five-state random walk. Exits on the right pay 1, on the left 0."""

    n = 5
    true_values = [(i + 1) / 6 for i in range(5)]

    def episode(self, rng):
        s, states = 2, []
        while 0 <= s < self.n:
            states.append(s)
            s += rng.choice((-1, 1))
        return states, 1.0 if s == self.n else 0.0


def _rms(values):
    return math.sqrt(sum((v - t) ** 2 for v, t in zip(values, RandomWalk.true_values)) / RandomWalk.n)


def td0_prediction(episodes: int = 100, alpha: float = 0.1, seed: int = 0):
    """Return (final values, RMS error after each episode)."""
    rng, env = random.Random(seed), RandomWalk()
    v, errors = [0.5] * env.n, []
    for _ in range(episodes):
        states, reward = env.episode(rng)
        for i, s in enumerate(states):
            if i + 1 < len(states):
                target = v[states[i + 1]]
            else:
                target = reward  # the exit
            v[s] += alpha * (target - v[s])
        errors.append(_rms(v))
    return v, errors


def mc_prediction(episodes: int = 100, alpha: float = 0.1, seed: int = 0):
    """Same task, but wait for the episode to end and move each value toward the return."""
    rng, env = random.Random(seed), RandomWalk()
    v, errors = [0.5] * env.n, []
    for _ in range(episodes):
        states, reward = env.episode(rng)
        for s in states:
            v[s] += alpha * (reward - v[s])
        errors.append(_rms(v))
    return v, errors


# -------------------------------------------------------------------- control

def _eps_greedy(q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(4)
    best = max(q[state])
    return rng.choice([a for a in range(4) if q[state][a] == best])


def _empty_q(env: GridWorld):
    return {s: [0.0] * 4 for s in env.states()}


def _learn(env, episodes, alpha, gamma, epsilon, seed, on_policy, max_steps=2000):
    rng, q, returns = random.Random(seed), _empty_q(env), []
    for _ in range(episodes):
        s, total, done, steps = env.start, 0.0, False, 0
        a = _eps_greedy(q, s, epsilon, rng)
        while not done and steps < max_steps:
            nxt, r, done = env.step(s, a)
            a2 = _eps_greedy(q, nxt, epsilon, rng)
            bootstrap = 0.0 if done else (q[nxt][a2] if on_policy else max(q[nxt]))
            q[s][a] += alpha * (r + gamma * bootstrap - q[s][a])
            s, a, total, steps = nxt, a2, total + r, steps + 1
        returns.append(total)
    return q, returns


def q_learning(env: GridWorld, episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
               epsilon: float = 0.1, seed: int = 0):
    """Off-policy TD control: learn the value of the best policy while exploring."""
    return _learn(env, episodes, alpha, gamma, epsilon, seed, on_policy=False)


def sarsa(env: GridWorld, episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
          epsilon: float = 0.1, seed: int = 0):
    """On-policy TD control: learn the value of the policy actually being followed."""
    return _learn(env, episodes, alpha, gamma, epsilon, seed, on_policy=True)


def greedy_path(env: GridWorld, q, max_steps: int = 200):
    """Follow the greedy policy. Returns (total reward, list of states, reached goal)."""
    s, total, path = env.start, 0.0, [env.start]
    for _ in range(max_steps):
        best = max(q[s])
        a = min(i for i in range(4) if q[s][i] == best)
        s, r, done = env.step(s, a)
        total += r
        path.append(s)
        if done:
            return total, path, True
    return total, path, False


def actor_critic(env: GridWorld, episodes: int = 300, alpha_critic: float = 0.2,
                 alpha_actor: float = 0.2, gamma: float = 1.0, seed: int = 0, max_steps: int = 500):
    """One-step actor-critic. The critic learns V, the actor learns action preferences,
    and both are driven by the same TD error. Returns (preferences, values, returns)."""
    rng = random.Random(seed)
    prefs = {s: [0.0] * 4 for s in env.states()}
    v = {s: 0.0 for s in env.states()}
    returns = []
    for _ in range(episodes):
        s, total, done, steps = env.start, 0.0, False, 0
        while not done and steps < max_steps:
            h = prefs[s]
            m = max(h)
            w = [math.exp(x - m) for x in h]
            z = sum(w)
            probs = [x / z for x in w]
            a = rng.choices(range(4), weights=probs)[0]
            nxt, r, done = env.step(s, a)
            delta = r + (0.0 if done else gamma * v[nxt]) - v[s]
            v[s] += alpha_critic * delta
            for b in range(4):
                prefs[s][b] += alpha_actor * delta * ((1.0 if b == a else 0.0) - probs[b])
            s, total, steps = nxt, total + r, steps + 1
        returns.append(total)
    return prefs, v, returns


# --------------------------------------------------------------- dopamine toy

def dopamine_conditioning(trials: int = 200, delay: int = 5, alpha: float = 0.2, gamma: float = 0.95):
    """A cue appears at an unpredictable time, and a reward arrives `delay` steps later.

    The state is the time since the cue, t = 0 .. delay - 1, and the reward is paid on
    the step after the last of them. Before the cue the value is fixed at zero, because
    the cue's arrival cannot be predicted. Returns, for each trial, the TD error at the
    moment the cue appears and at the moment the reward is paid.
    """
    v = [0.0] * delay
    history = []
    for _ in range(trials):
        delta_cue = gamma * v[0]  # jump from the unpredicted baseline (value 0) to the cue state
        for t in range(delay - 1):
            v[t] += alpha * (gamma * v[t + 1] - v[t])
        delta_reward = 1.0 - v[delay - 1]
        v[delay - 1] += alpha * delta_reward
        history.append({"cue": delta_cue, "reward": delta_reward})
    return history
