import unittest

from thought_machines.mentalizing import Room, Scene, goal_posterior, posterior_trace, sally_anne, sample_trajectory

LAYOUT = ["A.......B", ".........", "....S....", ".........", "....C...."]


class InversePlanningTests(unittest.TestCase):
    def setUp(self):
        self.room = Room(LAYOUT)

    def test_room_parses_goals_and_start(self):
        self.assertEqual(self.room.goals, {"A": (0, 0), "B": (0, 8), "C": (4, 4)})
        self.assertEqual(self.room.start, (2, 4))

    def test_prior_is_returned_with_no_evidence(self):
        post = goal_posterior(self.room, [], beta=2.0)
        for v in post.values():
            self.assertAlmostEqual(v, 1 / 3)

    def test_action_probabilities_sum_to_one_and_favour_progress(self):
        probs = self.room.action_probabilities(self.room.start, self.room.goals["C"], 2.0)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertEqual(probs.index(max(probs)), 2)  # down, toward C

    def test_posterior_normalises(self):
        traj = sample_trajectory(self.room, "B", 2.0, 5, seed=0)
        self.assertAlmostEqual(sum(goal_posterior(self.room, traj, 2.0).values()), 1.0)

    def test_a_few_steps_identify_the_goal(self):
        hits = total = 0
        for goal in "ABC":
            for seed in range(40):
                traj = sample_trajectory(self.room, goal, 2.0, 6, seed)
                post = goal_posterior(self.room, traj, 2.0)
                hits += max(post, key=post.get) == goal
                total += 1
        self.assertGreater(hits / total, 0.95)

    def test_more_evidence_helps(self):
        def accuracy(steps):
            hits = 0
            for goal in "ABC":
                for seed in range(40):
                    post = goal_posterior(self.room, sample_trajectory(self.room, goal, 2.0, steps, seed), 2.0)
                    hits += max(post, key=post.get) == goal
            return hits
        self.assertGreater(accuracy(4), accuracy(1))

    def test_random_agent_reveals_nothing(self):
        for seed in range(5):
            traj = sample_trajectory(self.room, "A", 0.0, 8, seed)
            post = goal_posterior(self.room, traj, 0.0)
            for v in post.values():
                self.assertAlmostEqual(v, 1 / 3)

    def test_trace_has_one_entry_per_step_plus_prior(self):
        traj = sample_trajectory(self.room, "A", 3.0, 4, seed=1)
        self.assertEqual(len(posterior_trace(self.room, traj, 3.0)), len(traj) + 1)

    def test_tied_goals_stay_tied_after_a_symmetric_first_move(self):
        post = goal_posterior(self.room, [(self.room.start, 0)], beta=3.0)  # up: A and B equally closer
        self.assertAlmostEqual(post["A"], post["B"])
        self.assertLess(post["C"], 0.05)

    def test_prior_matters_when_evidence_is_weak(self):
        post = goal_posterior(self.room, [], 2.0, prior={"A": 0.8, "B": 0.1, "C": 0.1})
        self.assertAlmostEqual(post["A"], 0.8)


class FalseBeliefTests(unittest.TestCase):
    def test_sally_looks_where_she_last_saw_it(self):
        s = sally_anne()
        self.assertEqual(s.reality("marble"), "box")
        self.assertEqual(s.predict_search("Sally", "marble", mentalizing=True), "basket")
        self.assertEqual(s.predict_search("Sally", "marble", mentalizing=False), "box")

    def test_true_belief_control_both_predictors_agree(self):
        s = sally_anne(sally_watches_the_move=True)
        self.assertEqual(s.predict_search("Sally", "marble", True), s.predict_search("Sally", "marble", False))

    def test_anne_knows_what_sally_believes(self):
        s = sally_anne()
        self.assertEqual(s.belief("Anne", "marble"), "box")
        self.assertEqual(s.belief_about_belief("Anne", "Sally", "marble"), "basket")

    def test_unseen_object_has_no_belief(self):
        s = Scene(["Sally"])
        self.assertIsNone(s.belief("Sally", "marble"))


if __name__ == "__main__":
    unittest.main()
