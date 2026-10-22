"""Regression checks for guidance selection and simulation bookkeeping."""

import unittest
from unittest.mock import patch

import numpy as np

import params as P
from pronav import pronav
from sim_assignment import sim


class AssignmentSimulationTests(unittest.TestCase):
    def test_defenders_choose_guidance_independently(self):
        # Robot 1 must use moving-ball PN even when Robot 3 has a fixed ghost.
        # Previously Robot 3's else branch overwrote Robot 1's command.
        ghosts = [None, np.array([400., 1000., 0.]),
                  np.array([400., -1000., 0.])]
        assignments = [(None, ghost, [], None) for ghost in ghosts]
        with patch("sim_assignment.get_cost_metrics", return_value=assignments) as planner:
            result = sim(plot=False, animate=False, verbose=False, horizon=P.Ts)
        self.assertEqual(planner.call_args.kwargs["selection_mode"], "slowest_symmetric")

        ball_start = np.array([1000., 0., np.pi])
        for i, (y, ghost) in enumerate(zip([100., 0., -100.], ghosts)):
            expected = pronav(
                np.array([0., y, 0.]), ball_start if ghost is None else ghost,
                30., ghost_speed=100. if ghost is None else 0.,
                max_acceleration=2 * P.gravity,
            )[0]
            self.assertAlmostEqual(result["controls"][i][1], expected)
        self.assertTrue(np.isnan(result["ghost_poses"][0][1]).all())
        np.testing.assert_allclose(result["ghost_poses"][2][1], ghosts[2])

    def test_no_feasible_ghost_still_records_a_complete_run(self):
        no_assignments = [(None, None, [], None)] * 3
        with patch("sim_assignment.get_cost_metrics", return_value=no_assignments):
            result = sim(plot=False, animate=False, verbose=False, horizon=0.04)

        self.assertIsNone(result["hit_robot"])
        self.assertIsNone(result["hit_time"])
        self.assertFalse(result["ball_reached_goal"])
        np.testing.assert_allclose(result["times"], [0., .01, .02, .03, .04])
        for i in range(3):
            self.assertEqual(result["robot_poses"][i].shape, (5, 3))
            self.assertTrue(np.isnan(result["ghost_poses"][i]).all())
            self.assertTrue(np.isfinite(result["controls"][i][1:]).all())
            self.assertAlmostEqual(result["closest_approach"][i],
                                   min(result["ranges"][i]))

    def test_replanning_handles_intervals_between_integration_steps(self):
        no_assignments = [(None, None, [], None)] * 3
        with patch("sim_assignment.get_cost_metrics",
                   return_value=no_assignments) as planner:
            result = sim(plot=False, animate=False, verbose=False,
                         horizon=0.07, update_interval=0.025)

        np.testing.assert_allclose(result["replan_times"], [0.03, 0.05])
        self.assertEqual(planner.call_count, 3)
        remaining = [call.args[-1] for call in planner.call_args_list]
        np.testing.assert_allclose(remaining, [.07, .04, .02])

    def test_selected_mode_is_forwarded_at_initialization_and_every_replan(self):
        for mode in ("slowest_symmetric", "median"):
            with self.subTest(mode=mode):
                with patch("sim_assignment.get_cost_metrics",
                           return_value=[(None, None, [], None)] * 3) as planner:
                    result = sim(plot=False, animate=False, verbose=False,
                                 horizon=0.07, update_interval=0.025,
                                 selection_mode=mode)
                self.assertEqual(result["selection_mode"], mode)
                self.assertEqual(planner.call_count, 3)
                for call in planner.call_args_list:
                    self.assertEqual(call.kwargs["selection_mode"], mode)


if __name__ == "__main__":
    unittest.main()
