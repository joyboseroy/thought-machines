"""Generic state-space search: BFS, DFS, uniform cost, greedy and A*.

A problem supplies `initial`, `is_goal(state)`, `successors(state)` yielding
(action, next_state, step_cost), and optionally `heuristic(state)`. Each search
reports how much work it did, which is the point: a heuristic changes the cost
of finding an answer, not what the answer is.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass


@dataclass
class SearchResult:
    strategy: str
    found: bool
    path: list
    cost: float
    expanded: int
    generated: int
    max_frontier: int


def _path(parent, state):
    actions = []
    while parent[state] is not None:
        state, action = parent[state]
        actions.append(action)
    return actions[::-1]


def search(problem, strategy: str = "astar", max_expansions: int | None = None) -> SearchResult:
    if strategy not in ("bfs", "dfs", "ucs", "greedy", "astar"):
        raise ValueError(f"unknown strategy {strategy!r}")
    start = problem.initial
    parent = {start: None}
    expanded = generated = max_frontier = 0

    def finish(found, state=None, cost=0.0):
        return SearchResult(strategy, found, _path(parent, state) if found else [], cost,
                            expanded, generated, max_frontier)

    if strategy in ("bfs", "dfs"):
        frontier = deque([(start, 0.0)])
        while frontier:
            max_frontier = max(max_frontier, len(frontier))
            state, g = frontier.popleft() if strategy == "bfs" else frontier.pop()
            expanded += 1
            if problem.is_goal(state):
                return finish(True, state, g)
            if max_expansions and expanded >= max_expansions:
                break
            for action, nxt, cost in problem.successors(state):
                if nxt not in parent:
                    parent[nxt] = (state, action)
                    generated += 1
                    frontier.append((nxt, g + cost))
        return finish(False)

    h = problem.heuristic if strategy in ("greedy", "astar") else (lambda s: 0)

    def priority(g, s):
        return {"ucs": g, "greedy": h(s), "astar": g + h(s)}[strategy]

    tie = itertools.count()
    heap = [(priority(0, start), next(tie), 0.0, start)]
    best_g, closed = {start: 0.0}, set()
    while heap:
        max_frontier = max(max_frontier, len(heap))
        _, _, g, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        expanded += 1
        if problem.is_goal(state):
            return finish(True, state, g)
        if max_expansions and expanded >= max_expansions:
            break
        for action, nxt, cost in problem.successors(state):
            ng = g + cost
            if nxt not in best_g or ng < best_g[nxt]:
                best_g[nxt] = ng
                parent[nxt] = (state, action)
                generated += 1
                heapq.heappush(heap, (priority(ng, nxt), next(tie), ng, nxt))
    return finish(False)
