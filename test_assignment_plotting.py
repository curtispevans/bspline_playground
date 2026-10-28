"""Checks that plots display recorded planner decisions and replan timing."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assignment_plotting import (DEFENDER_COLORS, UNASSIGNED_COLOR, assignment_at_time,
                                 assignment_summary, draw_assignment_costs,
                                 draw_assignment_paths)
from intercept import get_cost_metrics
from plotting_animation import animate_simulation, plot_simulation
from sim_assignment_unknown_heading import sim as sim_unknown_heading


def recorded_plan(costs, selection_mode="median", symmetric=True, robot_poses=None):
    """Make a real planner record with controlled interception predictions."""
    costs = np.asarray(costs, dtype=float)
    headings = np.linspace(np.pi / 2, 3 * np.pi / 2, costs.shape[1])
    robots = ([np.array([0., 100., 0.]), np.array([0., 0., 0.]),
               np.array([0., -100., 0.])] if robot_poses is None else robot_poses)
    if not symmetric:
        robots[1][2] = 0.2
    predictions = [
        (float(costs[i, j]), np.array([500., 80. * (i - 1), 0.]), None)
        if np.isfinite(costs[i, j]) else None
        for j in range(len(headings)) for i in range(len(robots))
    ]
    plan = {}
    with patch("intercept.find_intercept", side_effect=predictions):
        results = get_cost_metrics(
            robots, np.array([1000., 0., np.pi]), np.array([0., 0., np.pi]),
            30., 100., 19.62, 39.24, horizon=2.,
            terminal_thetas=headings, diagnostics=plan,
            selection_mode=selection_mode,
        )
    plan.update(time=0., effective_time=0.)
    return results, plan


class AssignmentPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_planner_partitions_headings_for_any_team_size(self):
        for count in (1, 2, 5, 12):
            with self.subTest(count=count):
                robots = [np.array([10., 30. * i, 0.1]) for i in range(count)]
                costs = np.full((count, 3), 20.)
                for j in range(3):
                    costs[j % count, j] = j + 1.
                results, plan = recorded_plan(costs, robot_poses=robots)
                self.assertEqual(len(results), count)
                self.assertEqual(plan["intercept_times"].shape, (count, 3))
                np.testing.assert_array_equal(plan["owners"], np.arange(3) % count)
                for i, result in enumerate(results):
                    owned = np.flatnonzero(plan["owners"] == i)
                    self.assertEqual(len(result[2]), len(owned))
                    if not len(owned):
                        self.assertEqual(result, (None, None, [], None))
                    else:
                        self.assertIn(plan["selected_indices"][i], owned)

    def test_symmetric_trio_does_not_override_a_larger_asymmetric_team(self):
        base = [np.array([0., y, 0.]) for y in [100., 0., -100.]]
        for extras in ([[0., 250., 0.]], [[0., 100., 0.]],
                       [[0., 250., 0.1], [0., -250., 0.]]):
            with self.subTest(extras=extras):
                robots = base + [np.array(pose) for pose in extras]
                costs = np.full((len(robots), 5), 30.)
                costs[0, 2], costs[1, 2] = 1., 9.
                _, plan = recorded_plan(costs, selection_mode="slowest_symmetric",
                                        robot_poses=robots)
                self.assertEqual(plan["owners"][2], 0)
                self.assertNotIn("symmetry", plan["selection_reasons"])

    def test_symmetric_five_defender_team_keeps_midpoint_override(self):
        robots = [np.array([0., y, 0.]) for y in [200., 100., 0., -100., -200.]]
        costs = np.full((5, 5), 30.)
        np.fill_diagonal(costs, 9.)
        costs[0, 2] = 1.
        _, plan = recorded_plan(costs, selection_mode="slowest_symmetric",
                                robot_poses=robots)
        self.assertEqual(plan["owners"][2], 2)
        self.assertEqual(plan["selected_indices"][2], 2)
        self.assertEqual(plan["selection_reasons"][2], "symmetry")

    def test_snapshots_and_replay_display_any_team_size(self):
        for count in (1, 2, 5, len(DEFENDER_COLORS) + 1):
            with self.subTest(count=count):
                times = np.arange(3) * 0.01
                robots = [np.array([[0., 10. * i, 0.], [1., 10. * i, 0.],
                                    [2., 10. * i, 0.]]) for i in range(count)]
                ball = np.tile([1000., 0., np.pi], (3, 1))
                ghosts = [np.tile([400., 10. * i, 0.], (3, 1)) for i in range(count)]
                controls = [np.full(3, i + 0.25) for i in range(count)]
                goal = np.array([0., 0., np.pi])

                snapshots = plot_simulation(
                    times, robots, ball, ghosts, [], goal[:2], None, show=False,
                )
                for ax in snapshots.axes:
                    defender_lines = [line for line in ax.lines
                                      if line.get_linewidth() == 2.2]
                    self.assertEqual(len(defender_lines), count)
                    for i, line in enumerate(defender_lines):
                        self.assertEqual(line.get_color(),
                                         DEFENDER_COLORS[i % len(DEFENDER_COLORS)])
                self.assertIn(f"Defender {count}",
                              [text.get_text() for text in snapshots.legends[0].get_texts()])

                fig, animation = animate_simulation(
                    times, robots, ball, ghosts, controls, goal, show=False,
                )
                slider = fig._simulation_widgets[0]
                for frame in (2, 0, 1):
                    slider.set_val(frame)
                    trails = [line for line in fig.axes[0].lines
                              if line.get_label().startswith("Defender ")
                              and line.get_label().endswith(" trail")]
                    self.assertEqual(len(trails), count)
                    for i, trail in enumerate(trails):
                        np.testing.assert_allclose(trail.get_xdata(), robots[i][:frame + 1, 0])
                        np.testing.assert_allclose(trail.get_ydata(), robots[i][:frame + 1, 1])
                    labels = ", ".join(f"D{i + 1}" for i in range(count))
                    ranges = ", ".join(f"{np.linalg.norm(poses[frame, :2] - ball[frame, :2]):.1f}"
                                       for poses in robots)
                    control_text = ", ".join(f"{values[frame]:+.2f}" for values in controls)
                    status = fig.texts[0].get_text()
                    self.assertIn(f"range [{labels}]: [{ranges}] m", status)
                    self.assertIn(f"normal accel [{labels}]: [{control_text}] m/s²", status)
                animation.event_source.stop()
                plt.close(fig)

    def test_modes_preserve_partition_and_choose_slowest_or_median(self):
        costs = [
            [7, 1, 5, 30, 2, 30, 30, 30, np.inf],
            [30, 30, 30, 9, 4, 8, 30, 30, np.inf],
            [30, 30, 30, 30, 30, 30, 6, 2, np.inf],
        ]
        _, median = recorded_plan(costs, selection_mode="median")
        _, slowest = recorded_plan(costs, selection_mode="slowest_symmetric")
        for key in ("owners", "eligible_mask", "intercept_times"):
            np.testing.assert_array_equal(median[key], slowest[key])
        np.testing.assert_array_equal(median["selected_indices"], [2, 5, 7])
        np.testing.assert_array_equal(slowest["selected_indices"], [0, 4, 6])
        self.assertEqual(slowest["selection_reasons"], ["slowest", "symmetry", "slowest"])
        # The centered robot chooses the midpoint even though its cost is lower.
        self.assertEqual(slowest["intercept_times"][1, 4], 4.)
        self.assertEqual(slowest["intercept_times"][1, 3], 9.)
        self.assertIn("midpoint", assignment_summary(slowest))
        self.assertNotIn("median", assignment_summary(slowest))
        _, ax = plt.subplots()
        draw_assignment_costs(ax, slowest)
        self.assertIn("slowest with symmetry", ax.get_title())

    def test_slowest_mode_uses_maximum_when_geometry_is_not_symmetric(self):
        _, plan = recorded_plan([
            [7, 1, 5, 30, 2, 30, 30, 30, np.inf],
            [30, 30, 30, 9, 4, 8, 30, 30, np.inf],
            [30, 30, 30, 30, 30, 30, 6, 2, np.inf],
        ], selection_mode="slowest_symmetric", symmetric=False)
        np.testing.assert_array_equal(plan["selected_indices"], [0, 3, 6])
        self.assertEqual(plan["selection_reasons"], ["slowest"] * 3)

    def test_unknown_selection_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            recorded_plan([[1, 8, 8], [8, 1, 8], [8, 8, 1]],
                          selection_mode="unsupported")

    def test_diagnostics_record_symmetric_ownership_and_selected_ghosts(self):
        results, plan = recorded_plan([
            [7, 1, 5, 30, 2, 30, 30, 30, np.inf],
            [30, 30, 30, 8, 9, 3, 30, 30, np.inf],
            [30, 30, 30, 30, 30, 30, 6, 2, np.inf],
        ])
        # The center owns theta=pi despite robot 1's lower cost at that sample.
        np.testing.assert_array_equal(plan["owners"], [0, 0, 0, 1, 1, 1, 2, 2, -1])
        np.testing.assert_array_equal(plan["selected_indices"], [2, 3, 7])
        for i, result in enumerate(results):
            j = plan["selected_indices"][i]
            self.assertEqual(plan["terminal_thetas"][j], result[3])
            self.assertEqual(plan["intercept_times"][i, j], result[0])
            np.testing.assert_array_equal(plan["selected_poses"][i], result[1])

    def test_missing_intercepts_are_drawn_as_unassigned(self):
        results, plan = recorded_plan(np.full((3, 3), np.inf))
        self.assertEqual(results, [(None, None, [], None)] * 3)
        np.testing.assert_array_equal(plan["owners"], [-1, -1, -1])
        fig, (ax, cost_ax) = plt.subplots(1, 2)
        artists = draw_assignment_paths(ax, plan)
        self.assertEqual(len(artists), 3)
        for artist in artists:
            self.assertEqual(artist.get_color(), UNASSIGNED_COLOR)
            self.assertEqual(artist.get_linestyle(), ":")
        draw_assignment_costs(cost_ax, plan)
        self.assertTrue(any("No feasible intercepts" in text.get_text()
                            for text in cost_ax.texts))
        fig.canvas.draw()

    def test_replay_uses_recorded_plan_and_can_step_back_across_replan(self):
        _, initial = recorded_plan([[1, 8, 8], [8, 1, 8], [8, 8, 1]])
        _, refreshed = recorded_plan([[80, 80, 10], [80, 10, 80], [10, 80, 80]])
        refreshed.update(time=0.02, effective_time=0.03)
        history = [initial, refreshed]
        self.assertIs(assignment_at_time(history, 0.02), initial)
        self.assertIs(assignment_at_time(history, 0.03), refreshed)
        times = np.arange(5) * 0.01
        robots = [np.tile(pose, (5, 1)) for pose in initial["robot_poses"]]
        ball = np.tile(initial["ball_pose"], (5, 1))
        ghosts = [np.full((5, 3), np.nan) for _ in robots]
        controls = [np.zeros(5) for _ in robots]
        with patch("plotting_animation.draw_assignment_paths",
                   wraps=draw_assignment_paths) as draw:
            fig, animation = animate_simulation(
                times, robots, ball, ghosts, controls, initial["goal"],
                assignment_history=history, show=False,
            )
            slider = fig._simulation_widgets[0]
            original_line_count = len(fig.axes[0].lines)
            self.assertLess(fig.axes[1].get_ylim()[1], 9.)
            slider.set_val(2)
            self.assertIs(draw.call_args.args[1], initial)
            slider.set_val(3)
            self.assertIs(draw.call_args.args[1], refreshed)
            self.assertIn("0.02 s", fig.axes[1].get_title())
            self.assertGreater(fig.axes[1].get_ylim()[1], 80.)
            slider.set_val(0)
            self.assertIs(draw.call_args.args[1], initial)
            self.assertIn("0.00 s", fig.axes[1].get_title())
            self.assertLess(fig.axes[1].get_ylim()[1], 9.)
            self.assertEqual(len(fig.axes[0].lines), original_line_count)
            animation.event_source.stop()

    def test_unknown_heading_simulation_passes_the_entire_bank_to_both_displays(self):
        headings = np.array([1.2, 2., np.pi, 4., 4.9])
        with patch("sim_assignment_unknown_heading.find_intercept", return_value=None), \
                patch("sim_assignment_unknown_heading.plot_simulation") as plot, \
                patch("sim_assignment_unknown_heading.animate_simulation",
                      return_value=(None, None)) as animate:
            result = sim_unknown_heading(
                candidate_headings=headings, true_terminal_heading=2.2,
                update_ghosts=False, verbose=False, plot_path=None,
            )

        paths = result["candidate_paths"]
        self.assertEqual(len(paths), len(headings))
        self.assertIs(plot.call_args.kwargs["candidate_paths"], paths)
        self.assertIs(animate.call_args.kwargs["candidate_paths"], paths)
        np.testing.assert_array_equal(result["candidate_headings"], headings)
        for path, heading in zip(paths, headings):
            self.assertTrue(np.isfinite(path).all())
            np.testing.assert_allclose(path[0, :2], [1000., 0.], atol=1e-9)
            np.testing.assert_allclose(path[-1, :2], [0., 0.], atol=1e-9)
            self.assertAlmostEqual(np.sin(path[-1, 2] - heading), 0.)
            self.assertAlmostEqual(np.cos(path[-1, 2] - heading), 1.)

    def test_candidate_paths_fit_snapshots_and_persist_when_scrubbing_replay(self):
        times = np.arange(5) * 0.01
        robots = [np.tile([0., y, 0.], (5, 1)) for y in [100., 0., -100.]]
        ball = np.tile([1000., 0., np.pi], (5, 1))
        ghosts = [np.full((5, 3), np.nan) for _ in robots]
        controls = [np.zeros(5) for _ in robots]
        goal = np.array([0., 0., np.pi])
        paths = [np.array([[1000., 0.], [600., y], [-200., y], [0., 0.]])
                 for y in [-700., 700.]]

        def candidate_collection(ax):
            collections = [artist for artist in ax.collections
                           if artist.get_label() == "Possible intruder trajectories"]
            self.assertEqual(len(collections), 1)
            np.testing.assert_allclose(collections[0].get_segments(), paths)
            self.assertLess(ax.get_xlim()[0], -200.)
            self.assertGreater(ax.get_xlim()[1], 1000.)
            self.assertLess(ax.get_ylim()[0], -700.)
            self.assertGreater(ax.get_ylim()[1], 700.)
            return collections[0]

        snapshots = plot_simulation(
            times, robots, ball, ghosts, [], goal[:2], None,
            candidate_paths=paths, show=False,
        )
        for ax in snapshots.axes:
            candidate_collection(ax)
        self.assertIn("Possible intruder trajectories",
                      [text.get_text() for text in snapshots.legends[0].get_texts()])

        fig, animation = animate_simulation(
            times, robots, ball, ghosts, controls, goal,
            candidate_paths=paths, show=False,
        )
        original = candidate_collection(fig.axes[0])
        slider = fig._simulation_widgets[0]
        for frame in (4, 2, 0):
            slider.set_val(frame)
            self.assertIs(candidate_collection(fig.axes[0]), original)
        self.assertIn("Possible intruder trajectories",
                      [text.get_text() for text in fig.legends[0].get_texts()])
        animation.event_source.stop()


if __name__ == "__main__":
    unittest.main()
