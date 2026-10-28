"""Behavior checks for the ball's maximum-acceleration / PN policy."""

import unittest
from unittest.mock import patch

import numpy as np

import params as P
from dynamics import SimpleDynamics
from path_manager import MaxAccelerationPronavPathManager
from pronav import pronav
from sim_assignment import sim


class BallPathManagerTests(unittest.TestCase):
    def test_maneuver_follows_the_minimum_radius_circle_in_both_directions(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                ball = SimpleDynamics(np.zeros(3), 100., P.max_acceleration)
                manager = MaxAccelerationPronavPathManager(
                    [1000., 0.], turn_direction=direction,
                )
                for _ in range(200):
                    command = manager.update(ball)
                    np.testing.assert_array_equal(command, [direction * P.max_acceleration])
                    ball.update(command)

                radius = ball.v0**2 / ball.max_acceleration
                angle = ball.max_acceleration * 2.0 / ball.v0
                expected = [radius * np.sin(angle),
                            direction * radius * (1 - np.cos(angle)),
                            direction * angle]
                np.testing.assert_allclose(ball.x, expected, atol=1e-8)
                self.assertIsNone(manager.switch_time)

                expected_command = pronav(
                    ball.x, [1000., 0., 0.], ball.v0,
                    max_acceleration=ball.max_acceleration,
                )
                np.testing.assert_allclose(manager.update(ball), expected_command)
                self.assertEqual(manager.path_state, "pronav")
                self.assertAlmostEqual(manager.switch_time, 2.0)

    def test_switch_uses_first_step_after_duration_and_never_restarts(self):
        ball = SimpleDynamics(np.zeros(3), 100., P.max_acceleration)
        manager = MaxAccelerationPronavPathManager(
            [1000., 0.], maneuver_duration=0.025,
        )
        for _ in range(3):
            ball.update(manager.update(ball))
            self.assertEqual(manager.path_state, "maneuver")
        ball.update(manager.update(ball))
        self.assertEqual(manager.path_state, "pronav")
        self.assertAlmostEqual(manager.switch_time, 0.03)
        # Passing the goal must not restart the maneuver.
        ball.x = np.array([1100., 0., 0.])
        manager.update(ball)
        self.assertEqual(manager.path_state, "pronav")
        self.assertAlmostEqual(manager.switch_time, 0.03)

    def test_zero_duration_starts_in_pn_with_configured_gain_and_saturation(self):
        for gain in (2., 5.):
            with self.subTest(gain=gain):
                ball = SimpleDynamics(np.array([0., 0., 0.2]), 100., 20.)
                manager = MaxAccelerationPronavPathManager(
                    [1000., 0.], maneuver_duration=0., navigation_constant=gain,
                )
                expected = pronav(
                    ball.x, [1000., 0., 0.], ball.v0,
                    navigation_constant=gain, max_acceleration=20.,
                )
                np.testing.assert_allclose(manager.update(ball), expected)
                self.assertEqual(manager.switch_time, 0.)
                self.assertEqual(manager.path_state, "pronav")

    def test_ball_reaches_goal_after_maneuver_for_left_and_right_turns(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                ball = SimpleDynamics(np.array([1000., 0., np.pi]), 100., P.max_acceleration)
                manager = MaxAccelerationPronavPathManager(
                    [0., 0.], turn_direction=direction,
                )
                for _ in range(3000):
                    command = manager.update(ball)
                    self.assertLessEqual(abs(command[0]), ball.max_acceleration)
                    if manager.done:
                        break
                    ball.update(command)
                self.assertTrue(manager.done)
                self.assertLess(np.linalg.norm(ball.x[:2]), 10.)
                np.testing.assert_array_equal(manager.update(ball), [0.])

    def test_goal_arrival_completes_during_either_phase(self):
        for duration in (0., 2.):
            with self.subTest(duration=duration):
                ball = SimpleDynamics(np.array([1., 0., 0.]), 100., P.max_acceleration)
                manager = MaxAccelerationPronavPathManager(
                    [0., 0.], maneuver_duration=duration,
                )
                np.testing.assert_array_equal(manager.update(ball), [0.])
                self.assertTrue(manager.done)
                self.assertEqual(manager.path_state, "complete")
                self.assertIsNone(manager.switch_time)

    def test_invalid_configuration_is_rejected(self):
        invalid = [
            {"goal_position": [0.]}, {"goal_position": [0., np.nan]},
            {"maneuver_duration": -1.}, {"maneuver_duration": np.inf},
            {"turn_direction": 0}, {"turn_direction": np.nan},
            {"navigation_constant": 0.}, {"navigation_constant": np.nan},
            {"goal_radius": 0.}, {"goal_radius": np.inf},
        ]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    MaxAccelerationPronavPathManager(
                        **{"goal_position": [0., 0.], **options},
                    )


class BallPolicySimulationTests(unittest.TestCase):
    def test_simulation_records_commands_and_actual_switch_time(self):
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 3):
            result = sim(
                plot=False, animate=False, verbose=False, horizon=0.07,
                ball_maneuver_duration=0.025, ball_turn_direction=-1,
                ball_navigation_constant=4.,
            )
        self.assertEqual(result["ball_policy"], "max_accel_pronav")
        self.assertAlmostEqual(result["ball_switch_time"], 0.03)
        np.testing.assert_allclose(result["ball_controls"][1:4], -P.max_acceleration)
        np.testing.assert_array_equal(
            result["ball_path_states"], ["initial"] + ["maneuver"] * 3 + ["pronav"] * 4,
        )
        for i in range(4, len(result["times"])):
            expected = pronav(
                result["ball_poses"][i - 1], [0., 0., 0.], 100.,
                navigation_constant=4., max_acceleration=P.max_acceleration,
            )[0]
            self.assertAlmostEqual(result["ball_controls"][i], expected)

    def test_pn_goal_has_no_terminal_heading_constraint(self):
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 3):
            result = sim(
                plot=False, animate=False, verbose=False, horizon=0.04,
                ball_maneuver_duration=0., terminal_heading=np.pi / 2,
            )
        # The ball already points at the goal, so PN must keep it straight.
        np.testing.assert_allclose(result["ball_controls"][1:], 0., atol=1e-10)
        np.testing.assert_allclose(result["ball_poses"][-1], [996., 0., np.pi], atol=1e-10)
        self.assertEqual(result["ball_switch_time"], 0.)

    def test_legacy_dubins_policy_is_available(self):
        with patch("sim_assignment.get_cost_metrics",
                   return_value=[(None, None, [], None)] * 3):
            result = sim(
                plot=False, animate=False, verbose=False, horizon=0.04,
                ball_policy="dubins", terminal_heading=np.pi,
            )
        self.assertEqual(result["ball_policy"], "dubins")
        self.assertIsNone(result["ball_switch_time"])
        np.testing.assert_allclose(result["ball_poses"][-1], [996., 0., np.pi], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
