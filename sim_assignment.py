"""Periodic ghost assignment with PN guidance and optional simulation replay."""

import numpy as np

import params as P
from dynamics import SimpleDynamics
from intercept import SELECTION_MODES, get_cost_metrics
from path_manager import PathManager
from plotting_animation import animate_simulation, plot_simulation
from pronav import pronav


def _pose_for_history(pose):
    """NaNs mark direct PN to the moving ball, with no stationary ghost."""
    return np.full(3, np.nan) if pose is None else np.asarray(pose).copy()


def sim(animate=True, plot=True, terminal_heading=4.422615269729036,
        update_interval=0.5, horizon=100.0, hit_radius=10.0,
        plot_path="sim_assignment.png", verbose=True, goal_radius=10.0,
        selection_mode="slowest_symmetric"):
    """Run the simple three-defender simulation and return its recorded data.

    Long intruder detours are removed from a dynamic terminal-heading bank.
    The defenders jointly divide the remaining headings according to
    which robot can intercept each heading first. ``slowest_symmetric`` chooses
    the slowest assigned intercept, with a midpoint heading for the centered
    robot in symmetric geometry. Set ``selection_mode="median"`` to choose
    the median intercept time in every region instead.
    Guidance switches to the moving ball inside the ball's minimum turn radius,
    or whenever the planner finds no ghost. Set ``plot=False, animate=False``
    for runs without figures. ``plot_path=None`` displays snapshots without
    saving them. See ``sim_assignment_algorithm.md`` for planning assumptions.
    """
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    if not np.all(np.isfinite(
            [terminal_heading, update_interval, horizon, hit_radius, goal_radius])):
        raise ValueError("Simulation parameters must be finite")
    if update_interval <= 0 or horizon < P.Ts or hit_radius <= 0 or goal_radius <= 0:
        raise ValueError("update_interval, hit_radius and goal_radius must be positive; "
                         "horizon must be at least P.Ts")

    def log(message):
        if verbose:
            print(message)

    ball_initial = np.array([1000., 0., np.pi])
    robot_initial = [np.array([0., 100., 0.]),
                     np.array([0., 0., 0.]),
                     np.array([0., -100., 0.])]
    target = np.array([0., 0., terminal_heading])
    v_robot, v_ball = 30., 100.
    a_max_robot, a_max_ball = 2 * P.gravity, P.max_acceleration
    ball_radius = v_ball**2 / a_max_ball

    robots = [SimpleDynamics(pose.copy(), v_robot, a_max_robot)
              for pose in robot_initial]
    ball = SimpleDynamics(ball_initial.copy(), v_ball, a_max_ball)
    ball_manager = PathManager(
        ball_radius, ball_initial[:2], ball_initial[2], target[:2], target[2],
    )

    assignment_history = []

    def assign_ghosts(remaining_time, plan_time=0.0):
        diagnostics = {}
        results = get_cost_metrics(
            [robot.x for robot in robots], ball.x, target,
            v_robot, v_ball, a_max_robot, a_max_ball, remaining_time,
            diagnostics=diagnostics,
            goal_radius=goal_radius,
            selection_mode=selection_mode,
        )
        if diagnostics:
            # At a replan instant the recorded poses still carry the old ghosts.
            diagnostics.update(time=plan_time,
                               effective_time=0.0 if not assignment_history
                               else plan_time + P.Ts)
            assignment_history.append(diagnostics)
        headings_deg = [None if res[3] is None else float(np.rad2deg(res[3]))
                        for res in results]
        log(f"Selected terminal headings (deg): {headings_deg}")
        return [res[1] for res in results]

    log(f"Selection mode: {selection_mode}")
    log(f"Terminal heading: {np.rad2deg(terminal_heading):.6f} deg")
    log("Computing initial ghost targets for each robot...")
    ghosts = assign_ghosts(horizon)
    times = [0.0]
    robot_history = [[robot.x.copy()] for robot in robots]
    ball_history = [ball.x.copy()]
    ghost_history = [[_pose_for_history(ghost)] for ghost in ghosts]
    control_history = [[np.nan] for _ in robots]
    replan_times = []
    next_replan = update_interval
    hit_robot = None
    hit_target = False
    t = 0.0

    log("Starting simulation...")
    for step in range(int(np.floor(horizon / P.Ts + 1e-9))):
        t = step * P.Ts
        # Replan on the first integration step at/after the requested time.
        if t + 1e-9 >= next_replan:
            ghosts = assign_ghosts(horizon - t, t)
            replan_times.append(t)
            log(f"t={t:.2f} s: refreshed ghost targets")
            next_replan = (np.floor((t + 1e-9) / update_interval) + 1) * update_interval

        controls = []
        active_ghosts = []
        for robot, ghost in zip(robots, ghosts):
            direct_pn = (ghost is None or
                         np.linalg.norm(robot.x[:2] - ball.x[:2]) < 1.25*ball_radius)
            # Choose each robot's mode independently, at every control step.
            controls.append(pronav(
                robot.x, ball.x if direct_pn else ghost, v_robot,
                ghost_speed=v_ball if direct_pn else 0.0,
                navigation_constant=3, max_acceleration=a_max_robot,
            ))
            active_ghosts.append(None if direct_pn else ghost)

        ball.update(ball_manager.update(ball))
        for robot, control in zip(robots, controls):
            robot.update(control)
        t = (step + 1) * P.Ts

        times.append(t)
        ball_history.append(ball.x.copy())
        for i, (robot, ghost, control) in enumerate(
                zip(robots, active_ghosts, controls)):
            robot_history[i].append(robot.x.copy())
            ghost_history[i].append(_pose_for_history(ghost))
            control_history[i].append(float(control[0]))

        for i, robot in enumerate(robots, 1):
            if np.linalg.norm(robot.x[:2] - ball.x[:2]) < hit_radius:
                hit_robot = i
                break
        if hit_robot is not None:
            break
        if np.linalg.norm(ball.x[:2] - target[:2]) < goal_radius:
            hit_target = True
            break

    if hit_robot is not None:
        outcome = f"Defender {hit_robot} intercepted at t={t:.2f} s"
    elif hit_target:
        outcome = f"Ball reached the goal at t={t:.2f} s"
    else:
        outcome = f"Time limit reached at t={t:.2f} s"
    log(outcome)

    times = np.asarray(times)
    robot_poses = [np.asarray(history) for history in robot_history]
    ball_poses = np.asarray(ball_history)
    ghost_poses = [np.asarray(history) for history in ghost_history]
    controls = [np.asarray(history) for history in control_history]
    ranges = [np.linalg.norm(poses[:, :2] - ball_poses[:, :2], axis=1)
              for poses in robot_poses]
    closest_indices = [int(np.argmin(values)) for values in ranges]
    if plot:
        plot_simulation(
            times, robot_poses, ball_poses, ghost_poses,
            [(instant, None) for instant in replan_times],
            target[:2], hit_robot, plot_path,
            title="Periodic ghost assignment with PN guidance",
            final_status=outcome,
            assignment_history=assignment_history,
        )
        if plot_path is not None:
            log(f"Saved snapshots to {plot_path}")

    replay = None
    if animate:
        replay = animate_simulation(
            times, robot_poses, ball_poses, ghost_poses, controls, target,
            assignment_history=assignment_history,
        )
    return {
        "selection_mode": selection_mode,
        "hit_robot": hit_robot,
        "hit_time": t if hit_robot is not None else None,
        "ball_reached_goal": hit_target,
        "outcome": outcome,
        "times": times,
        "robot_poses": robot_poses,
        "ball_poses": ball_poses,
        "ghost_poses": ghost_poses,
        "controls": controls,
        "ranges": ranges,
        "closest_approach": [float(values[index])
                             for values, index in zip(ranges, closest_indices)],
        "closest_approach_times": [float(times[index]) for index in closest_indices],
        "replan_times": np.asarray(replan_times),
        "assignment_history": assignment_history,
        "replay": replay,
    }


if __name__ == "__main__":
    sim(terminal_heading=np.radians(290), verbose=True,
        # selection_mode="slowest_symmetric")
        selection_mode="median")  # Or "median".
