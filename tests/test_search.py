import unittest

from thought_machines.search.core import search
from thought_machines.search.problems import (DEFAULT_MAZE, EightPuzzle, GridMaze,
                                              MissionariesCannibals, TowerOfHanoi)


class SearchTests(unittest.TestCase):
    def test_hanoi_optimal_is_2_to_the_n_minus_1(self):
        for n in (2, 3, 4):
            self.assertEqual(search(TowerOfHanoi(n), "bfs").cost, 2 ** n - 1)

    def test_missionaries_and_cannibals_optimal_is_11(self):
        self.assertEqual(search(MissionariesCannibals(3), "bfs").cost, 11)

    def test_astar_matches_bfs_cost_with_less_work(self):
        for seed in range(4):
            p = EightPuzzle.scrambled(22, seed=seed)
            bfs, astar = search(p, "bfs"), search(p, "astar")
            self.assertEqual(astar.cost, bfs.cost)
            self.assertLessEqual(astar.expanded, bfs.expanded)

    def test_greedy_can_be_suboptimal_but_finds_a_solution(self):
        p = EightPuzzle.scrambled(24, seed=2)
        self.assertTrue(search(p, "greedy").found)
        self.assertGreaterEqual(search(p, "greedy").cost, search(p, "astar").cost)

    def test_path_replays_to_the_goal(self):
        p = EightPuzzle.scrambled(15, seed=5)
        r = search(p, "astar")
        state = p.initial
        for action in r.path:
            state = next(t for a, t, _ in p.successors(state) if a == action)
        self.assertTrue(p.is_goal(state))

    def test_maze_and_unreachable_goal(self):
        self.assertTrue(search(GridMaze(DEFAULT_MAZE), "astar").found)
        walled = GridMaze(["S#G"])
        self.assertFalse(search(walled, "bfs").found)

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            search(TowerOfHanoi(2), "magic")


if __name__ == "__main__":
    unittest.main()
