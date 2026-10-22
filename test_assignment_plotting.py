"""Checks that plots display recorded planner decisions and replan timing."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from assignment_plotting import (UNASSIGNED_COLOR, assignment_at_time,
                                 assignment_summary, draw_assignment_costs,
                                 draw_assignment_paths)
from intercept import get_cost_metrics
from plotting_animation import animate_simulation


def recorded_plan(costs, selection_mode="median", symmetric=True):
    """Make a real planner record with controlled interception predictions."""
    costs = np.asarray(costs, dtype=float)
    headings = np.linspace(np.pi / 2, 3 * np.pi / 2, costs.shape[1])
    robots = [np.array([0., 100., 0.]), np.array([0., 0., 0.]),
              np.array([0., -100., 0.])]
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


if __name__ == "__main__":
    unittest.main()
