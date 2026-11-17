"""Classic problems for the search engine."""

from __future__ import annotations

import random


class EightPuzzle:
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    _MOVES = {"up": -3, "down": 3, "left": -1, "right": 1}

    def __init__(self, initial):
        self.initial = tuple(initial)

    def is_goal(self, s):
        return s == self.GOAL

    def successors(self, s):
        z = s.index(0)
        r, c = divmod(z, 3)
        for name, d in self._MOVES.items():
            if (name == "up" and r == 0) or (name == "down" and r == 2):
                continue
            if (name == "left" and c == 0) or (name == "right" and c == 2):
                continue
            t = list(s)
            t[z], t[z + d] = t[z + d], t[z]
            yield name, tuple(t), 1

    def heuristic(self, s):
        total = 0
        for i, v in enumerate(s):
            if v:
                g = v - 1
                total += abs(i // 3 - g // 3) + abs(i % 3 - g % 3)
        return total

    @classmethod
    def scrambled(cls, moves: int = 20, seed: int = 0):
        """A solvable instance made by random-walking away from the goal."""
        rng = random.Random(seed)
        p = cls(cls.GOAL)
        s, prev = cls.GOAL, None
        for _ in range(moves):
            options = [t for _, t, _ in p.successors(s) if t != prev]
            prev, s = s, rng.choice(options)
        return cls(s)


class TowerOfHanoi:
    """State: tuple giving the peg (0, 1, 2) of each disk, smallest disk first."""

    def __init__(self, n: int):
        self.n = n
        self.initial = (0,) * n
        self.goal = (2,) * n

    def is_goal(self, s):
        return s == self.goal

    def successors(self, s):
        for disk in range(self.n):
            src = s[disk]
            if any(s[k] == src for k in range(disk)):
                continue  # something smaller sits on top
            for dst in range(3):
                if dst == src or any(s[k] == dst for k in range(disk)):
                    continue
                t = list(s)
                t[disk] = dst
                yield f"disk {disk + 1}: {src}->{dst}", tuple(t), 1

    def heuristic(self, s):
        return 0


class MissionariesCannibals:
    """State: (missionaries on left, cannibals on left, boat on left). Boat holds two."""

    def __init__(self, n: int = 3):
        self.n = n
        self.initial = (n, n, True)

    def is_goal(self, s):
        return s[0] == 0 and s[1] == 0

    def _safe(self, m, c):
        n = self.n
        return (m == 0 or m >= c) and (n - m == 0 or n - m >= n - c)

    def successors(self, s):
        m, c, boat = s
        sign = -1 if boat else 1
        for dm, dc in ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1)):
            nm, nc = m + sign * dm, c + sign * dc
            if 0 <= nm <= self.n and 0 <= nc <= self.n and self._safe(nm, nc):
                yield f"move {dm}M {dc}C", (nm, nc, not boat), 1

    def heuristic(self, s):
        return 0


class GridMaze:
    def __init__(self, rows):
        self.rows = rows
        self.initial = self.goal = None
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch == "S":
                    self.initial = (r, c)
                elif ch == "G":
                    self.goal = (r, c)

    def is_goal(self, s):
        return s == self.goal

    def successors(self, s):
        r, c = s
        for name, dr, dc in (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.rows) and 0 <= nc < len(self.rows[nr]) and self.rows[nr][nc] != "#":
                yield name, (nr, nc), 1

    def heuristic(self, s):
        return abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1])


DEFAULT_MAZE = [
    "S...#.......#...",
    ".##.#.#####.#.#.",
    ".#..#.#...#...#.",
    ".#.##.#.#.#####.",
    ".#....#.#.......",
    ".######.#######.",
    "........#.....G.",
    ".######...#####.",
]
