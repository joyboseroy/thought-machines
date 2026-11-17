from .core import search
from .problems import DEFAULT_MAZE, EightPuzzle, GridMaze, MissionariesCannibals, TowerOfHanoi


def _table(title, problem, strategies=("bfs", "dfs", "ucs", "greedy", "astar"), max_expansions=None):
    print(f"\n{title}")
    print(f"{'strategy':>8} {'cost':>6} {'expanded':>9} {'generated':>10}")
    for s in strategies:
        r = search(problem, s, max_expansions=max_expansions)
        cost = f"{r.cost:.0f}" if r.found else "none"
        print(f"{s:>8} {cost:>6} {r.expanded:>9} {r.generated:>10}")


def main():
    print("Same answers, very different amounts of thinking.")
    _table("8-puzzle (scrambled 22 moves, seed 3)", EightPuzzle.scrambled(22, seed=3),
           strategies=("bfs", "ucs", "greedy", "astar"))
    _table("Tower of Hanoi, 4 disks (optimal is 15)", TowerOfHanoi(4), strategies=("bfs", "ucs"))
    _table("Missionaries and cannibals, 3 each (optimal is 11)", MissionariesCannibals(3),
           strategies=("bfs", "dfs", "ucs"))
    _table("Maze 8x16", GridMaze(DEFAULT_MAZE))


if __name__ == "__main__":
    main()
