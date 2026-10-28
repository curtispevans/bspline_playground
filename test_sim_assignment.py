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

    def test_timing_separates_planner_compute_and_display_calls(self):
        clock = [100.]
        durations = iter([0.02, 0.03, 0.04])

        def planner(*args, diagnostics, **kwargs):
            clock[0] += next(durations)
            diagnostics["selection_mode"] = kwargs["selection_mode"]
            return [(None, None, [], None)] * 3

        def plot(*args, **kwargs):
            clock[0] += 100.

        def animate(*args, **kwargs):
            clock[0] += 200.
            return None, None

        with patch("sim_assignment.perf_counter", side_effect=lambda: clock[0]), \
                patch("sim_assignment.get_cost_metrics", side_effect=planner), \
                patch("sim_assignment.plot_simulation", side_effect=plot), \
                patch("sim_assignment.animate_simulation", side_effect=animate):
            result = sim(verbose=False, horizon=0.07, update_interval=0.025,
                         plot_path=None)

        timing = result["timing"]
        np.testing.assert_allclose(timing["planning_seconds"], [0.02, 0.03, 0.04])
        np.testing.assert_allclose(
            [plan["compute_seconds"] for plan in result["assignment_history"]],
            timing["planning_seconds"],
        )
        self.assertAlmostEqual(timing["simulation_seconds"], 0.09)
        self.assertAlmostEqual(timing["planning_total_seconds"], 0.09)
        self.assertAlmostEqual(timing["initial_planning_seconds"], 0.02)
        self.assertAlmostEqual(timing["replan_mean_seconds"], 0.035)
        self.assertAlmostEqual(timing["replan_max_seconds"], 0.04)
        self.assertAlmostEqual(timing["realtime_factor"], 0.07 / 0.09)
        self.assertAlmostEqual(timing["plot_seconds"], 100.)
        self.assertAlmostEqual(timing["animation_seconds"], 200.)

    def test_timing_handles_a_run_without_replanning_or_figures(self):
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 3):
            result = sim(plot=False, animate=False, verbose=False, horizon=P.Ts)

        timing = result["timing"]
        self.assertEqual(len(timing["planning_seconds"]), 1)
        self.assertIsNone(timing["replan_mean_seconds"])
        self.assertIsNone(timing["replan_max_seconds"])
        self.assertEqual(timing["plot_seconds"], 0.)
        self.assertEqual(timing["animation_seconds"], 0.)
        self.assertGreater(timing["simulation_seconds"], 0.)
        self.assertGreaterEqual(timing["simulation_seconds"], timing["planning_total_seconds"])


if __name__ == "__main__":
    unittest.main()
