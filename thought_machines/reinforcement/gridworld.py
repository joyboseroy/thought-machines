"""A small deterministic grid world used by the agent tiers.

Layout characters: `#` wall, `S` start, `G` goal, `C` cliff, `.` open floor.
States are (row, column). Actions are 0 up, 1 right, 2 down, 3 left. Walking into a
wall or off the edge leaves the agent where it is. Stepping onto a cliff cell costs
`cliff_reward` and sends the agent back to the start without ending the episode.
"""

from __future__ import annotations

from collections import deque

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left

# Sutton and Barto's cliff walking task: the short path runs along the cliff edge.
CLIFF = ["............",
         "............",
         "............",
         "SCCCCCCCCCCG"]

SMALL_ROOM = ["S...",
              "....",
              "....",
              "...G"]

# A maze with two walls to get around. The shortest path from S to G is 14 steps.
DYNA_MAZE = [".......#G",
             "..#....#.",
             "S.#....#.",
             "..#....#.",
             ".........",
             "........."]


class GridWorld:
    def __init__(self, layout, step_reward: float = -1.0, goal_reward: float = -1.0,
                 cliff_reward: float = -100.0):
        self.layout = list(layout)
        self.rows, self.cols = len(self.layout), len(self.layout[0])
        self.step_reward, self.goal_reward, self.cliff_reward = step_reward, goal_reward, cliff_reward
        self.walls, self.cliffs = set(), set()
        self.start = self.goal = None
        for r, line in enumerate(self.layout):
            if len(line) != self.cols:
                raise ValueError("layout rows must have equal length")
            for c, ch in enumerate(line):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "C":
                    self.cliffs.add((r, c))
                elif ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("layout needs an S and a G")

    def states(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.walls]

    def step(self, state, action):
        """Return (next_state, reward, done)."""
        dr, dc = ACTIONS[action]
        nxt = (state[0] + dr, state[1] + dc)
        if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols) or nxt in self.walls:
            nxt = state
        if nxt in self.cliffs:
            return self.start, self.cliff_reward, False
        if nxt == self.goal:
            return nxt, self.goal_reward, True
        return nxt, self.step_reward, False

    def shortest_path_length(self):
        """Fewest steps from start to goal, by breadth-first search on the true world."""
        seen = {self.start: 0}
        queue = deque([self.start])
        while queue:
            s = queue.popleft()
            for a in range(4):
                nxt, _, done = self.step(s, a)
                if done:
                    return seen[s] + 1
                if nxt not in seen:  # a cliff step lands back on the start, already seen
                    seen[nxt] = seen[s] + 1
                    queue.append(nxt)
        return None
