"""Mentalizing, part 1: reading goals from behaviour by inverting a model of planning.

Bennett's fourth breakthrough, in early primates, is mentalizing: modelling the minds
of others. One formal account (Baker, Saxe and Tenenbaum, 2009) treats an observed
agent as roughly rational, so that it picks actions that lead efficiently to what it
wants. Run that model forward and you can predict behaviour from a goal. Run it
backward with Bayes' rule and you can infer a goal from behaviour:

    P(goal | actions)  proportional to  P(goal) * product over steps of P(action | state, goal)

The planner here is noisily rational (Boltzmann): moving closer to the goal is more
probable by a factor exp(beta) per unit of distance gained. beta = 0 is an agent that
moves at random and reveals nothing. Large beta is an agent that never wastes a move.
"""

from __future__ import annotations

import math
import random
from collections import deque

from ..reinforcement.gridworld import ACTIONS


class Room:
    """A grid with walls. `A`, `B`, `C`... are candidate goals and `S` is the start."""

    def __init__(self, layout):
        self.layout = list(layout)
        self.rows, self.cols = len(self.layout), len(self.layout[0])
        self.walls, self.goals, self.start = set(), {}, None
        for r, line in enumerate(self.layout):
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "S":
                    self.start = (r, c)
                elif ch.isalpha():
                    self.goals[ch] = (r, c)
        self._dist = {}

    def move(self, state, action):
        dr, dc = ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols) or nxt in self.walls:
            return state
        return nxt

    def distances(self, target):
        """Steps from every reachable cell to `target`, by breadth-first search."""
        if target not in self._dist:
            dist, queue = {target: 0}, deque([target])
            while queue:
                s = queue.popleft()
                for a in range(4):
                    nxt = self.move(s, a)
                    if nxt not in dist:
                        dist[nxt] = dist[s] + 1
                        queue.append(nxt)
            self._dist[target] = dist
        return self._dist[target]

    def action_probabilities(self, state, goal_cell, beta):
        """The rational-agent model: P(action | state, goal)."""
        dist = self.distances(goal_cell)
        logits = [-beta * dist[self.move(state, a)] for a in range(4)]
        m = max(logits)
        w = [math.exp(x - m) for x in logits]
        z = sum(w)
        return [x / z for x in w]


def goal_posterior(room: Room, trajectory, beta: float, prior=None):
    """Posterior over the room's goals given a list of (state, action) pairs."""
    names = sorted(room.goals)
    log_post = {g: math.log((prior or {}).get(g, 1 / len(names))) for g in names}
    for state, action in trajectory:
        for g in names:
            log_post[g] += math.log(room.action_probabilities(state, room.goals[g], beta)[action])
    m = max(log_post.values())
    w = {g: math.exp(v - m) for g, v in log_post.items()}
    z = sum(w.values())
    return {g: w[g] / z for g in names}


def posterior_trace(room: Room, trajectory, beta: float, prior=None):
    """The posterior after 0, 1, 2, ... observed steps."""
    return [goal_posterior(room, trajectory[:t], beta, prior) for t in range(len(trajectory) + 1)]


def sample_trajectory(room: Room, goal: str, beta: float, steps: int, seed: int = 0):
    """Simulate a noisily rational agent heading for `goal`. Returns (state, action) pairs."""
    rng, state, out = random.Random(seed), room.start, []
    for _ in range(steps):
        if state == room.goals[goal]:
            break
        probs = room.action_probabilities(state, room.goals[goal], beta)
        action = rng.choices(range(4), weights=probs)[0]
        out.append((state, action))
        state = room.move(state, action)
    return out
