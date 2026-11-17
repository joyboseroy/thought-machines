import statistics as st
import unittest

from thought_machines.reinforcement import DYNA_MAZE, GridWorld
from thought_machines.simulating import (DynaQ, LearnedModelProblem, latent_learning, learning_curve,
                                         plan_by_search)


def maze():
    return GridWorld(DYNA_MAZE, step_reward=0.0, goal_reward=1.0)


class DynaTests(unittest.TestCase):
    def test_planning_cuts_the_real_steps_needed(self):
        env = maze()
        plain = st.mean(st.mean(learning_curve(env, 0, 30, seed=s)) for s in range(8))
        dyna = st.mean(st.mean(learning_curve(env, 50, 30, seed=s)) for s in range(8))
        self.assertLess(dyna, 0.5 * plain)

    def test_second_episode_is_already_short_with_planning(self):
        env = maze()
        second = st.mean(learning_curve(env, 50, 2, seed=s)[1] for s in range(8))
        self.assertLess(second, 150)

    def test_zero_planning_steps_is_plain_q_learning(self):
        agent = DynaQ(maze(), n_planning=0, seed=0)
        agent.run_episode()
        # With no planning, only the visited state-action pairs can have non-zero values.
        nonzero = sum(1 for qs in agent.q.values() for x in qs if x != 0.0)
        self.assertLessEqual(nonzero, len(agent.model))

    def test_the_model_matches_the_world(self):
        env = maze()
        agent = DynaQ(env, n_planning=5, seed=2)
        agent.run_episode()
        for (s, a), (r, nxt, done) in agent.model.items():
            self.assertEqual(env.step(s, a), (nxt, r, done))

    def test_runs_are_reproducible(self):
        self.assertEqual(learning_curve(maze(), 10, 5, seed=3), learning_curve(maze(), 10, 5, seed=3))


class LatentLearningTests(unittest.TestCase):
    def test_a_learned_map_makes_the_first_rewards_count_immediately(self):
        env = maze()
        with_model = [latent_learning(env, 50, explore_steps=3000, episodes=6, seed=s) for s in range(6)]
        model_free = [latent_learning(env, 0, explore_steps=3000, episodes=6, seed=s) for s in range(6)]
        late = lambda runs: st.mean(st.mean(r[2:]) for r in runs)  # noqa: E731
        self.assertLess(late(with_model), 40)
        self.assertGreater(late(model_free), 150)

    def test_the_environment_is_restored(self):
        env = maze()
        latent_learning(env, 5, explore_steps=200, episodes=1, seed=0)
        self.assertEqual((env.step_reward, env.goal_reward), (0.0, 1.0))


class SearchOverModelTests(unittest.TestCase):
    def test_no_route_before_the_goal_has_been_seen(self):
        self.assertIsNone(plan_by_search(DynaQ(maze(), seed=0)))

    def test_search_over_the_learned_model_finds_a_shortest_route_that_works(self):
        env = maze()
        for seed in range(3):
            agent = DynaQ(env, n_planning=50, seed=seed)
            for _ in range(10):
                agent.run_episode()
            path = plan_by_search(agent, "bfs")
            self.assertEqual(len(path), env.shortest_path_length())
            state, done = env.start, False
            for a in path:  # execute it in the real world
                state, _, done = env.step(state, a)
            self.assertTrue(done)

    def test_problem_exposes_only_what_was_experienced(self):
        agent = DynaQ(maze(), seed=0)
        agent.step(agent.env.start)
        problem = LearnedModelProblem(agent)
        self.assertEqual(len(list(problem.successors(agent.env.start))), 1)


if __name__ == "__main__":
    unittest.main()
