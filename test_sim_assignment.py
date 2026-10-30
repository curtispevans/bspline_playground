"""Regression checks for guidance selection and simulation bookkeeping."""

import unittest
from unittest.mock import patch

import numpy as np

import params as P
from pronav import pronav
from sim_assignment import sim


class AssignmentSimulationTests(unittest.TestCase):
    def test_gif_can_be_saved_without_opening_a_replay(self):
        figure = object()
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 3), \
                patch("sim_assignment.animate_simulation", return_value=(figure, None)) as replay, \
                patch("sim_assignment.plt.close") as close:
            result = sim(plot=False, animate=False, verbose=False, horizon=P.Ts,
                         gif_path="replay.gif", gif_fps=10, gif_speed=2.)
        self.assertIsNone(result["replay"])
        self.assertFalse(replay.call_args.kwargs["show"])
        self.assertEqual(replay.call_args.kwargs["gif_path"], "replay.gif")
        self.assertEqual(replay.call_args.kwargs["gif_fps"], 10)
        self.assertEqual(replay.call_args.kwargs["gif_speed"], 2.)
        close.assert_called_once_with(figure)

    def test_invalid_gif_timing_is_rejected_before_planning(self):
        for name in ("gif_fps", "gif_speed"):
            for value in (0., -1., np.inf, np.nan):
                with self.subTest(setting=name, value=value):
                    with patch("sim_assignment.get_cost_metrics") as planner:
                        with self.assertRaisesRegex(ValueError, name):
                            sim(plot=False, animate=False, verbose=False, horizon=P.Ts,
                                gif_path="replay.gif", **{name: value})
                        planner.assert_not_called()

    def test_defender_count_and_start_generate_a_centered_formation(self):
        cases = [(1, [-20.]), (np.int64(2), [0., -40.]),
                 (4, [40., 0., -40., -80.]), (5, [60., 20., -20., -60., -100.])]
        for count, expected_y in cases:
            with self.subTest(count=count):
                with patch("sim_assignment.get_cost_metrics",
                           return_value=[(None, None, [], None)] * count) as planner:
                    result = sim(num_defenders=count, defender_start=(25., -20.),
                                 defender_spacing=40., plot=False, animate=False,
                                 verbose=False, horizon=0.03, update_interval=0.02)
                starts = np.array([poses[0] for poses in result["robot_poses"]])
                np.testing.assert_allclose(starts[:, 0], 25.)
                np.testing.assert_allclose(starts[:, 1], expected_y)
                np.testing.assert_allclose(starts[:, 2], 0.)
                np.testing.assert_allclose(planner.call_args_list[0].args[0], starts)
                self.assertEqual(planner.call_count, 2)
                self.assertEqual(len(result["controls"]), count)

    def test_zero_spacing_starts_all_defenders_at_the_given_position(self):
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 4):
            result = sim(num_defenders=4, defender_start=(20., 30.),
                         defender_spacing=0., plot=False, animate=False,
                         verbose=False, horizon=P.Ts)
        np.testing.assert_allclose([poses[0] for poses in result["robot_poses"]],
                                   [[20., 30., 0.]] * 4)

    def test_explicit_starting_poses_work_with_a_matching_defender_count(self):
        starts = [[10., 20., 0.1], [30., 40., -0.2]]
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 2):
            result = sim(num_defenders=2, robot_initial=starts,
                         defender_start=(50., 60.), defender_spacing=10.,
                         plot=False, animate=False, verbose=False, horizon=P.Ts)
        np.testing.assert_allclose([poses[0] for poses in result["robot_poses"]], starts)

    def test_defender_count_must_match_explicit_starting_poses(self):
        with patch("sim_assignment.get_cost_metrics") as planner:
            with self.assertRaisesRegex(ValueError, "num_defenders must match"):
                sim(num_defenders=3, robot_initial=[[0., 0., 0.]],
                    plot=False, animate=False, verbose=False, horizon=P.Ts)
            planner.assert_not_called()

    def test_invalid_formation_settings_are_rejected_before_planning(self):
        invalid = {
            "num_defenders": [0, -1, 1.5, 2., True, np.bool_(False), "3", np.nan],
            "defender_start": [None, [], [1.], [1., 2., 3.], [[1., 2.]],
                               [np.nan, 0.], [0., np.inf], ["bad", 0.]],
            "defender_spacing": [-1., np.nan, np.inf, [100.], "bad"],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(setting=name, value=value):
                    with patch("sim_assignment.get_cost_metrics") as planner:
                        with self.assertRaisesRegex(ValueError, name):
                            sim(**{name: value}, plot=False, animate=False,
                                verbose=False, horizon=P.Ts)
                        planner.assert_not_called()

    def test_custom_teams_keep_every_defender_through_replanning(self):
        for count in (1, 2, 5, 12):
            with self.subTest(count=count):
                starts = np.array([[10 * i, 100 - 40 * i, 0.1 * i]
                                   for i in range(count)])
                original = starts.copy()

                def planner(robot_poses, *args, diagnostics, **kwargs):
                    diagnostics["robot_poses"] = np.array(robot_poses).copy()
                    # Alternate fixed-ghost and direct-ball guidance.
                    return [(None, np.array([500., 1000., 0.]) if i % 2 == 0
                             else None, [], None) for i in range(len(robot_poses))]

                with patch("sim_assignment.get_cost_metrics", side_effect=planner) as plan:
                    result = sim(robot_initial=starts, plot=False, animate=False,
                                 verbose=False, horizon=0.07, update_interval=0.025)

                np.testing.assert_array_equal(starts, original)
                self.assertEqual(plan.call_count, 3)
                np.testing.assert_allclose(result["replan_times"], [0.03, 0.05])
                np.testing.assert_allclose(result["assignment_history"][0]["robot_poses"],
                                           starts)
                for key in ("robot_poses", "ghost_poses", "controls", "ranges",
                            "closest_approach", "closest_approach_times"):
                    self.assertEqual(len(result[key]), count)
                for i in range(count):
                    self.assertEqual(result["robot_poses"][i].shape, (8, 3))
                    self.assertEqual(result["ghost_poses"][i].shape, (8, 3))
                    np.testing.assert_allclose(result["robot_poses"][i][0], starts[i])
                    self.assertTrue(np.isfinite(result["controls"][i][1:]).all())
                    target = (np.array([500., 1000., 0.]) if i % 2 == 0 else
                              np.array([1000., 0., np.pi]))
                    expected = pronav(
                        starts[i], target, 30., ghost_speed=0. if i % 2 == 0 else 100.,
                        max_acceleration=2 * P.gravity,
                    )[0]
                    self.assertAlmostEqual(result["controls"][i][1], expected)
                for plan_record, instant in zip(result["assignment_history"], [0., .03, .05]):
                    index = int(round(instant / P.Ts))
                    np.testing.assert_allclose(
                        plan_record["robot_poses"],
                        [poses[index] for poses in result["robot_poses"]],
                    )

    def test_integer_pose_lists_use_floating_point_dynamics(self):
        starts = [[0, 25, 0], [10, -25, 1]]
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 2):
            result = sim(robot_initial=starts, plot=False, animate=False,
                         verbose=False, horizon=P.Ts)
        self.assertEqual(starts, [[0, 25, 0], [10, -25, 1]])
        self.assertAlmostEqual(result["robot_poses"][0][1, 0], 0.3, places=4)
        self.assertTrue(np.issubdtype(result["robot_poses"][0].dtype, np.floating))

    def test_interception_reports_input_order_beyond_the_third_defender(self):
        starts = [[0., y, 0.] for y in [200., 100., 0., -100.]]
        starts.append([995., 0., np.pi])
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * len(starts)):
            result = sim(robot_initial=starts, plot=False, animate=False,
                         verbose=False, horizon=0.1)
        self.assertEqual(result["hit_robot"], 5)
        self.assertEqual(result["hit_time"], P.Ts)
        self.assertIn("Defender 5 intercepted", result["outcome"])

    def test_invalid_team_poses_are_rejected_before_planning(self):
        invalid = [[], [0., 0., 0.], [[0., 0.]], [[0., 0., 0., 0.]],
                   [[[0., 0., 0.]]], [[0., 0., 0.], [0., 0.]],
                   [[np.nan, 0., 0.]], [[0., 0., np.inf]], [["bad", 0., 0.]]]
        for starts in invalid:
            with self.subTest(starts=starts):
                with patch("sim_assignment.get_cost_metrics") as planner:
                    with self.assertRaisesRegex(ValueError, "robot_initial"):
                        sim(robot_initial=starts, plot=False, animate=False, verbose=False)
                    planner.assert_not_called()

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
