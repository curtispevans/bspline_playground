import numpy as np
import params as P
from dynamics import SimpleDynamics
from intercept_fixed import get_cost_metric
from path_manager import PathManager
from pronav import pronav
import matplotlib.pyplot as plt
from plotting_animation import plot_simulation, animate_simulation


def _pose_for_history(pose):
    """Store an unavailable intercept pose as NaNs for animation playback."""
    if pose is None:
        return np.full(3, np.nan)
    return np.asarray(pose, dtype=float).copy()


def sim(animate=True):
    """Assign fixed PN ghosts on the actual ball path and replay the match."""
    ball_initial = np.array([1000., 0., np.pi])
    robot_initial = [
        np.array([0., 100., 0.]),
        np.array([0., 0., 0.]),
        np.array([0., -100., 0.]),
    ]
    v_robot, v_ball = 30., 100.
    a_max_robot, a_max_ball = 2 * P.gravity, P.max_acceleration
    horizon, hit_radius = 100., 10.
    target = np.array([0., 0., 4.422615269729036])

    robots = [SimpleDynamics(pose.copy(), v_robot, a_max_robot)
              for pose in robot_initial]
    ball = SimpleDynamics(ball_initial.copy(), v_ball, a_max_ball)
    ball_path_manager = PathManager(
        v_ball**2 / a_max_ball, ball_initial[:2], ball_initial[2],
        target[:2], target[2],
    )

    # Each cost calculation uses the same stationary-ghost PN law used below.
    # Keep the actual terminal heading: it defines the ball trajectory.
    print("Computing PN ghost assignments for the actual ball path...")
    assignments = [
        get_cost_metric(
            pose, ball_initial, target, v_robot, v_ball,
            a_max_robot, a_max_ball, horizon, hit_radius=hit_radius,
        )
        for pose in robot_initial
    ]
    ghosts = []
    for index, (hit_time, ghost_pose, ghost_time, terminal_theta) in enumerate(assignments, 1):
        ghosts.append(None if ghost_pose is None else ghost_pose.copy())
        if ghost_pose is None:
            print(f"Robot {index}: no PN ghost found on this sampled path.")
        else:
            print(f"Robot {index}: ghost={ghost_pose[:2].round(2)}, "
                  f"predicted hit={hit_time:.2f} s, "
                  f"ball at ghost={ghost_time:.2f} s, "
                  f"terminal theta={terminal_theta:.6f} rad")

    sim_times = [0.0]
    robot_history = [[robot.x.copy()] for robot in robots]
    ball_history = [ball.x.copy()]
    ghost_history = [[_pose_for_history(ghost)] for ghost in ghosts]
    control_history = [[np.nan] for _ in robots]
    hit_robot = None
    hit_target = False
    t = 0.0

    for step in range(int(horizon / P.Ts)):
        controls = []
        for robot, ghost in zip(robots, ghosts):
            if ghost is None:
                # A robot with no feasible PN assignment remains unassigned.
                control = np.array([0.0])
            else:
                control = pronav(
                    robot.x, ghost, robot_speed=v_robot, ghost_speed=0.0,
                    navigation_constant=3, max_acceleration=a_max_robot,
                )
            controls.append(control)

        ball.update(ball_path_manager.update(ball))
        for robot, control in zip(robots, controls):
            robot.update(control)
        t = (step + 1) * P.Ts

        sim_times.append(t)
        ball_history.append(ball.x.copy())
        for i, (robot, ghost, control) in enumerate(zip(robots, ghosts, controls)):
            robot_history[i].append(robot.x.copy())
            ghost_history[i].append(_pose_for_history(ghost))
            control_history[i].append(float(control[0]))

        for i, robot in enumerate(robots):
            if np.linalg.norm(robot.x[:2] - ball.x[:2]) < hit_radius:
                hit_robot = i + 1
                print(f"Robot {hit_robot} hit the ball at t={t:.2f} s.")
                break
        if hit_robot is not None:
            break
        if np.linalg.norm(ball.x[:2] - target[:2]) < 5.0:
            hit_target = True
            print(f"Ball reached its goal at t={t:.2f} s.")
            break

    robot_poses = [np.asarray(history) for history in robot_history]
    ball_poses = np.asarray(ball_history)
    if animate:
        animate_simulation(
            np.asarray(sim_times), robot_poses, ball_poses,
            [np.asarray(history) for history in ghost_history],
            [np.asarray(history) for history in control_history], target,
        )
    return {
        "hit_robot": hit_robot,
        "hit_time": t if hit_robot is not None else None,
        "ball_reached_goal": hit_target,
        "assignments": assignments,
        "robot_poses": robot_poses,
        "ball_poses": ball_poses,
    }


if __name__ == "__main__":
    sim()
