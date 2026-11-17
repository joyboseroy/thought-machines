import statistics as st
import unittest

from thought_machines.reinforcement import (CLIFF, DYNA_MAZE, SMALL_ROOM, GridWorld, actor_critic,
                                            dopamine_conditioning, greedy_path, mc_prediction,
                                            q_learning, sarsa, td0_prediction)
from thought_machines.reinforcement.td import RandomWalk


class GridWorldTests(unittest.TestCase):
    def test_known_shortest_paths(self):
        self.assertEqual(GridWorld(CLIFF).shortest_path_length(), 13)
        self.assertEqual(GridWorld(SMALL_ROOM).shortest_path_length(), 6)
        self.assertEqual(GridWorld(DYNA_MAZE, 0.0, 1.0).shortest_path_length(), 14)

    def test_walls_and_edges_block_movement(self):
        env = GridWorld(SMALL_ROOM)
        self.assertEqual(env.step((0, 0), 0)[0], (0, 0))  # up off the edge
        self.assertEqual(env.step((0, 0), 3)[0], (0, 0))  # left off the edge
        maze = GridWorld(DYNA_MAZE, 0.0, 1.0)
        self.assertEqual(maze.step((2, 1), 1)[0], (2, 1))  # right into a wall

    def test_cliff_sends_the_agent_home_without_ending(self):
        env = GridWorld(CLIFF)
        state, reward, done = env.step((3, 0), 1)
        self.assertEqual((state, reward, done), (env.start, -100.0, False))

    def test_reaching_the_goal_ends_the_episode(self):
        env = GridWorld(SMALL_ROOM)
        self.assertEqual(env.step((3, 2), 1), ((3, 3), -1.0, True))

    def test_layout_needs_start_and_goal(self):
        with self.assertRaises(ValueError):
            GridWorld(["...", "..."])


class PredictionTests(unittest.TestCase):
    def test_td_beats_constant_step_monte_carlo_on_the_random_walk(self):
        td = st.mean(td0_prediction(100, 0.1, s)[1][-1] for s in range(60))
        mc = st.mean(mc_prediction(100, 0.1, s)[1][-1] for s in range(60))
        self.assertLess(td, 0.08)
        self.assertGreater(mc, 1.5 * td)

    def test_true_values_of_the_walk(self):
        self.assertEqual([round(x * 6) for x in RandomWalk.true_values], [1, 2, 3, 4, 5])


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(CLIFF)

    def test_q_learning_finds_the_shortest_path_along_the_cliff(self):
        for seed in range(3):
            q, _ = q_learning(self.env, episodes=500, seed=seed)
            total, path, ok = greedy_path(self.env, q)
            self.assertTrue(ok)
            self.assertEqual(len(path) - 1, 13)
            self.assertEqual(total, -13.0)

    def test_sarsa_takes_the_safe_path_the_textbook_describes(self):
        for seed in range(3):
            q, _ = sarsa(self.env, episodes=2000, seed=seed)
            total, path, ok = greedy_path(self.env, q)
            self.assertTrue(ok)
            self.assertEqual(len(path) - 1, 17)
            self.assertNotIn(3, [r for r, _ in path[1:-1]])  # never walks the cliff row

    def test_sarsa_does_better_while_learning_because_it_respects_exploration(self):
        _, q_returns = q_learning(self.env, episodes=1000, seed=1)
        _, s_returns = sarsa(self.env, episodes=1000, seed=1)
        self.assertGreater(st.mean(s_returns[-100:]), st.mean(q_returns[-100:]) + 10)

    def test_runs_are_reproducible(self):
        a = q_learning(self.env, episodes=50, seed=4)[1]
        b = q_learning(self.env, episodes=50, seed=4)[1]
        self.assertEqual(a, b)

    def test_actor_critic_learns_the_shortest_route(self):
        room = GridWorld(SMALL_ROOM)
        for seed in range(3):
            prefs, _, returns = actor_critic(room, episodes=300, seed=seed)
            total, _, ok = greedy_path(room, prefs)
            self.assertTrue(ok)
            self.assertEqual(total, -6.0)
            self.assertGreater(st.mean(returns[-20:]), st.mean(returns[:20]))


class DopamineTests(unittest.TestCase):
    def test_prediction_error_moves_from_the_reward_to_the_cue(self):
        h = dopamine_conditioning(trials=300, delay=5, gamma=0.95)
        self.assertAlmostEqual(h[0]["reward"], 1.0)  # a total surprise the first time
        self.assertEqual(h[0]["cue"], 0.0)
        self.assertLess(abs(h[-1]["reward"]), 1e-3)  # fully predicted after learning
        self.assertAlmostEqual(h[-1]["cue"], 0.95 ** 5, places=3)  # surprise now at the cue

    def test_omitted_reward_value_stays_consistent(self):
        # The reward error never rises again once learned, and the cue error never falls back.
        h = dopamine_conditioning(trials=300)
        self.assertTrue(all(abs(x["reward"]) < 0.05 for x in h[150:]))
        self.assertTrue(all(x["cue"] > 0.7 for x in h[150:]))


if __name__ == "__main__":
    unittest.main()
