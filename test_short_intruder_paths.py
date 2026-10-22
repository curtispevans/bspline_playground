"""Regression checks for short heading branches and terminal-turn geometry."""

import unittest
from unittest.mock import patch

import numpy as np

import params as P
from dynamics import SimpleDynamics
from intercept import (_approach_terminal_thetas, _default_terminal_thetas,
                       _short_intruder_paths, get_cost_metrics)
from path_manager import PathManager, get_dubins_parameters


class ShortIntruderPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.radius = 100.**2 / P.max_acceleration
        cls.goal = np.array([0., 0., np.pi / 2])
        cls.robots = [np.array([0., 100., 0.]), np.array([0., 0., 0.]),
                      np.array([0., -100., 0.])]
        ball = SimpleDynamics(np.array([1000., 0., np.pi]), 100., P.max_acceleration)
        manager = PathManager(cls.radius, ball.x[:2], ball.x[2],
                              cls.goal[:2], cls.goal[2])
        cls.states = {0: ball.x.copy()}
        for step in range(1, 901):
            ball.update(manager.update(ball))
            if step in (201, 600, 900):
                cls.states[step] = ball.x.copy()

    def candidates(self, ball, goal=None):
        goal = self.goal if goal is None else goal
        headings = _approach_terminal_thetas(ball, goal, self.radius)
        return headings, _short_intruder_paths(ball, goal, headings, self.radius)

    def test_initial_heading_bank_is_preserved(self):
        headings, (eligible, _, _, _) = self.candidates(self.states[0])
        np.testing.assert_allclose(headings, _default_terminal_thetas(np.pi))
        self.assertTrue(eligible.all())

    def test_long_candidates_at_2_01_are_never_evaluated_for_interception(self):
        diagnostics = {}
        with patch("intercept.find_intercept",
                   return_value=(1., np.zeros(3), None)) as predict:
            results = get_cost_metrics(
                self.robots, self.states[201], self.goal, 30., 100.,
                2 * P.gravity, P.max_acceleration, diagnostics=diagnostics,
            )
        eligible = diagnostics["eligible_mask"]
        lengths = diagnostics["intruder_path_lengths"]
        self.assertTrue(eligible.any())
        self.assertTrue((~eligible).any())
        self.assertLess(lengths[eligible].max(), 1100.)
        self.assertGreater(lengths[~eligible].min(), 2000.)
        self.assertEqual(predict.call_count, 3 * np.count_nonzero(eligible))
        self.assertTrue(np.isinf(diagnostics["intercept_times"][:, ~eligible]).all())
        for i, result in enumerate(results):
            if result[0] is not None:
                self.assertTrue(eligible[diagnostics["selected_indices"][i]])

    def test_short_terminal_heading_survives_outside_the_old_heading_bank(self):
        ball = self.states[600]
        self.assertGreater(_default_terminal_thetas(ball[2]).min(), np.pi / 2)
        headings, (eligible, lengths, _, _) = self.candidates(ball)
        j = int(np.argmin(np.abs(headings - np.pi / 2)))
        self.assertAlmostEqual(headings[j], np.pi / 2, places=5)
        self.assertTrue(eligible[j])
        self.assertLess(lengths[j], 600.)

    def test_terminal_turn_remains_a_single_arc(self):
        ball = np.array([self.radius, 0., np.pi / 2])
        goal = np.array([0., self.radius, np.pi])
        headings, (eligible, lengths, _, _) = self.candidates(ball, goal)
        self.assertEqual(np.count_nonzero(eligible), 1)
        theta = headings[eligible][0]
        self.assertAlmostEqual(theta, np.pi)
        params = get_dubins_parameters(ball[:2], ball[2], goal[:2], theta, self.radius)
        self.assertAlmostEqual(params[-1], 0.)
        self.assertAlmostEqual(lengths[eligible][0], self.radius * np.pi / 2)

    def test_intercept_search_ends_when_short_forecast_reaches_its_goal(self):
        diagnostics = {}
        with patch("intercept.find_intercept", return_value=None) as predict:
            get_cost_metrics(
                self.robots, self.states[900], self.goal, 30., 100.,
                2 * P.gravity, P.max_acceleration, diagnostics=diagnostics,
            )
        j = np.flatnonzero(diagnostics["eligible_mask"])[0]
        arrival = diagnostics["intruder_path_lengths"][j] / 100.
        self.assertLess(arrival, 3.)
        self.assertEqual(predict.call_count, 3)
        for call in predict.call_args_list:
            self.assertAlmostEqual(call.args[-1], arrival)

    def test_no_short_path_returns_no_ghosts_instead_of_choosing_a_detour(self):
        diagnostics = {}
        with patch("intercept.find_intercept") as predict:
            results = get_cost_metrics(
                self.robots, self.states[900], self.goal, 30., 100.,
                2 * P.gravity, P.max_acceleration, diagnostics=diagnostics,
                goal_radius=0.,
            )
        self.assertFalse(diagnostics["eligible_mask"].any())
        self.assertEqual(results, [(None, None, [], None)] * 3)
        predict.assert_not_called()

    def test_observed_terminal_turn_reaches_goal_region_without_new_maneuver(self):
        diagnostics = {}
        with patch("intercept.find_intercept", return_value=None):
            get_cost_metrics(
                self.robots, self.states[900], self.goal, 30., 100.,
                2 * P.gravity, P.max_acceleration, diagnostics=diagnostics,
            )
        eligible = diagnostics["eligible_mask"]
        self.assertEqual(np.count_nonzero(eligible), 1)
        j = np.flatnonzero(eligible)[0]
        self.assertTrue(diagnostics["terminal_turn_mask"][j])
        goal = diagnostics["candidate_goals"][j]
        self.assertLess(np.linalg.norm(goal[:2] - self.goal[:2]), 10.)
        params = get_dubins_parameters(self.states[900][:2], self.states[900][2],
                                       goal[:2], goal[2], self.radius)
        self.assertAlmostEqual(params[-1], 0.)
        self.assertAlmostEqual(params[-2], 0.)
        self.assertLess(diagnostics["intruder_path_lengths"][j], 300.)

    def test_reflected_90_and_270_scenarios_keep_matching_short_paths(self):
        ball = self.states[201].copy()
        headings, (eligible, lengths, _, _) = self.candidates(ball)
        ball[1] *= -1
        ball[2] = 2 * np.pi - ball[2]
        reflected_goal = np.array([0., 0., 3 * np.pi / 2])
        reflected, (accepted, reflected_lengths, _, _) = self.candidates(ball, reflected_goal)
        np.testing.assert_allclose(np.sort(2 * np.pi - headings[eligible]),
                                   reflected[accepted], atol=1e-8)
        np.testing.assert_allclose(np.sort(lengths[eligible]),
                                   np.sort(reflected_lengths[accepted]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
