"""Periodic ghost assignment with PN guidance and optional simulation replay."""

from time import perf_counter

import numpy as np

import params as P
from dynamics import SimpleDynamics
from intercept import SELECTION_MODES, get_cost_metrics
from path_manager import MaxAccelerationPronavPathManager, PathManager
from plotting_animation import animate_simulation, plot_simulation
from pronav import pronav


def _pose_for_history(pose):
    """NaNs mark direct PN to the moving ball, with no stationary ghost."""
    return np.full(3, np.nan) if pose is None else np.asarray(pose).copy()


def sim(animate=True, plot=True, terminal_heading=4.422615269729036,
        update_interval=0.5, horizon=100.0, hit_radius=10.0,
        plot_path="sim_assignment.png", verbose=True, goal_radius=10.0,
        selection_mode="slowest_symmetric", ball_policy="max_accel_pronav",
        ball_maneuver_duration=2.0, ball_turn_direction=1,
        ball_navigation_constant=3.0, robot_initial=None, *,
        num_defenders=None, defender_start=(0., 0.), defender_spacing=100.):
    """Run a defender-team simulation and return its recorded data.

    Set ``num_defenders`` (default 3) and ``defender_start=(x, y)`` to generate
    a vertical formation centered at that position. ``defender_spacing`` sets
    the distance between neighbors (default 100 m, zero allowed). Generated
    defenders face heading zero and are ordered from highest to lowest y.
    ``robot_initial`` is a nonempty (N, 3) array-like of defender starting poses
    ``[x, y, heading]``, with headings in radians, for individual positions and
    headings instead of the generated formation. Its row count sets the team
    size; an explicit ``num_defenders`` must match. All defenders share the same
    speed and acceleration limit. Defender IDs in the results and plots follow
    the input order, starting at 1.

    Long intruder detours are removed from a dynamic terminal-heading bank.
    The defenders jointly divide the remaining headings according to
    which robot can intercept each heading first. ``slowest_symmetric`` chooses
    the slowest assigned intercept, with a midpoint heading for the centered
    robot in symmetric geometry. Set ``selection_mode="median"`` to choose
    the median intercept time in every region instead.
    By default the ball turns left at maximum normal acceleration for two
    seconds, then uses PN to the goal position. Configure this with
    ``ball_maneuver_duration``, ``ball_turn_direction`` (+1 left, -1 right),
    and ``ball_navigation_constant``. Duration zero starts directly in PN.
    ``terminal_heading`` controls only the legacy ``ball_policy="dubins"``.
    Defender forecasts continue to use the bank of hypothetical Dubins paths;
    they do not receive the ball's actual maneuver duration or turn direction.
    Guidance switches to the moving ball inside the ball's minimum turn radius,
    or whenever the planner finds no ghost. Set ``plot=False, animate=False``
    for runs without figures. ``plot_path=None`` displays snapshots without
    saving them. See ``sim_assignment_algorithm.md`` for planning assumptions.
    Wall-clock measurements are returned in ``timing`` and printed when
    verbose. Compute time excludes figures; display-call times include any
    wait for windows to close. ``timing['planning_seconds']`` contains the
    initial assignment followed by each replan, in seconds.
    """
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    if ball_policy not in ("max_accel_pronav", "dubins"):
        raise ValueError("ball_policy must be 'max_accel_pronav' or 'dubins'")
    if not np.all(np.isfinite(
            [terminal_heading, update_interval, horizon, hit_radius, goal_radius])):
        raise ValueError("Simulation parameters must be finite")
    if update_interval <= 0 or horizon < P.Ts or hit_radius <= 0 or goal_radius <= 0:
        raise ValueError("update_interval, hit_radius and goal_radius must be positive; "
                         "horizon must be at least P.Ts")
    if num_defenders is not None and (
            isinstance(num_defenders, (bool, np.bool_))
            or not isinstance(num_defenders, (int, np.integer)) or num_defenders < 1):
        raise ValueError("num_defenders must be a positive integer")
    if robot_initial is None:
        count = 3 if num_defenders is None else num_defenders
        try:
            start = np.asarray(defender_start, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("defender_start must be a finite (x, y) position") from exc
        if start.shape != (2,) or not np.isfinite(start).all():
            raise ValueError("defender_start must be a finite (x, y) position")
        try:
            spacing = np.asarray(defender_spacing, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("defender_spacing must be finite and nonnegative") from exc
        if spacing.ndim != 0 or not np.isfinite(spacing) or spacing < 0:
            raise ValueError("defender_spacing must be finite and nonnegative")
        robot_initial = np.zeros((count, 3))
        robot_initial[:, :2] = start
        robot_initial[:, 1] += (0.5 * (count - 1) - np.arange(count)) * spacing
    pose_error = ("robot_initial must be a nonempty (N, 3) array of finite "
                  "[x, y, heading] poses")
    try:
        robot_initial = np.array(robot_initial, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(pose_error) from exc
    if (robot_initial.ndim != 2 or robot_initial.shape[0] == 0
            or robot_initial.shape[1] != 3 or not np.isfinite(robot_initial).all()):
        raise ValueError(pose_error)
    if num_defenders is not None and len(robot_initial) != num_defenders:
        raise ValueError("num_defenders must match the number of robot_initial poses")

    simulation_started = perf_counter()

    def log(message):
        if verbose:
            print(message)

    ball_initial = np.array([1000., 0., np.pi])
    target = np.array([0., 0., terminal_heading])
    v_robot, v_ball = 30., 100.
    a_max_robot, a_max_ball = 2 * P.gravity, P.max_acceleration
    ball_radius = v_ball**2 / a_max_ball

    robots = [SimpleDynamics(pose.copy(), v_robot, a_max_robot)
              for pose in robot_initial]
    ball = SimpleDynamics(ball_initial.copy(), v_ball, a_max_ball)
    if ball_policy == "max_accel_pronav":
        ball_manager = MaxAccelerationPronavPathManager(
            target[:2], maneuver_duration=ball_maneuver_duration,
            turn_direction=ball_turn_direction,
            navigation_constant=ball_navigation_constant, goal_radius=goal_radius,
        )
    else:
        ball_manager = PathManager(
            ball_radius, ball_initial[:2], ball_initial[2], target[:2], target[2],
        )

    assignment_history = []
    planning_seconds = []

    def assign_ghosts(remaining_time, plan_time=0.0):
        diagnostics = {}
        planning_started = perf_counter()
        results = get_cost_metrics(
            [robot.x for robot in robots], ball.x, target,
            v_robot, v_ball, a_max_robot, a_max_ball, remaining_time,
            diagnostics=diagnostics,
            goal_radius=goal_radius,
            selection_mode=selection_mode,
        )
        elapsed = perf_counter() - planning_started
        planning_seconds.append(elapsed)
        if diagnostics:
            # At a replan instant the recorded poses still carry the old ghosts.
            diagnostics.update(time=plan_time,
                               effective_time=0.0 if not assignment_history
                               else plan_time + P.Ts,
                               compute_seconds=elapsed)
            assignment_history.append(diagnostics)
        headings_deg = [None if res[3] is None else float(np.rad2deg(res[3]))
                        for res in results]
        log(f"Selected terminal headings (deg): {headings_deg}")
        log(f"t={plan_time:.2f} s: assignment computed in {elapsed * 1000:.2f} ms")
        return [res[1] for res in results]

    log(f"Selection mode: {selection_mode}")
    log(f"Ball policy: {ball_policy}")
    if ball_policy == "max_accel_pronav":
        direction = "left" if ball_turn_direction == 1 else "right"
        log(f"Ball maneuver: maximum {direction} acceleration for "
            f"{ball_maneuver_duration:.2f} s, then PN (N={ball_navigation_constant:g})")
    else:
        log(f"Terminal heading: {np.rad2deg(terminal_heading):.6f} deg")
    log("Computing initial ghost targets for each robot...")
    ghosts = assign_ghosts(horizon)
    times = [0.0]
    robot_history = [[robot.x.copy()] for robot in robots]
    ball_history = [ball.x.copy()]
    ghost_history = [[_pose_for_history(ghost)] for ghost in ghosts]
    control_history = [[np.nan] for _ in robots]
    ball_control_history = [np.nan]
    ball_path_states = ["initial"]
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

        ball_control = ball_manager.update(ball)
        ball.update(ball_control)
        for robot, control in zip(robots, controls):
            robot.update(control)
        t = (step + 1) * P.Ts

        times.append(t)
        ball_history.append(ball.x.copy())
        ball_control_history.append(float(ball_control[0]))
        ball_path_states.append(ball_manager.path_state)
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
    simulation_seconds = perf_counter() - simulation_started
    planning_seconds = np.asarray(planning_seconds)
    replan_seconds = planning_seconds[1:]
    timing = {
        "simulation_seconds": simulation_seconds,
        "planning_seconds": planning_seconds,
        "planning_total_seconds": float(planning_seconds.sum()),
        "initial_planning_seconds": float(planning_seconds[0]),
        "replan_mean_seconds": float(replan_seconds.mean()) if replan_seconds.size else None,
        "replan_max_seconds": float(replan_seconds.max()) if replan_seconds.size else None,
        "realtime_factor": t / simulation_seconds if simulation_seconds > 0 else np.inf,
        "plot_seconds": 0.0,
        "animation_seconds": 0.0,
    }
    log(f"Compute timing: {simulation_seconds:.3f} s wall for {t:.2f} s simulated "
        f"({timing['realtime_factor']:.2f}x real time; excludes figures)")
    log(f"  Initial assignment: {timing['initial_planning_seconds'] * 1000:.2f} ms; "
        f"total planning: {timing['planning_total_seconds']:.3f} s")
    if replan_seconds.size:
        log(f"  Replans ({replan_seconds.size}): "
            f"mean {timing['replan_mean_seconds'] * 1000:.2f} ms, "
            f"max {timing['replan_max_seconds'] * 1000:.2f} ms")

    if plot:
        plot_started = perf_counter()
        plot_simulation(
            times, robot_poses, ball_poses, ghost_poses,
            [(instant, None) for instant in replan_times],
            target[:2], hit_robot, plot_path,
            title="Periodic ghost assignment with PN guidance",
            final_status=outcome,
            assignment_history=assignment_history,
        )
        timing["plot_seconds"] = perf_counter() - plot_started
        if plot_path is not None:
            log(f"Saved snapshots to {plot_path}")

    replay = None
    if animate:
        animation_started = perf_counter()
        replay = animate_simulation(
            times, robot_poses, ball_poses, ghost_poses, controls, target,
            assignment_history=assignment_history,
        )
        timing["animation_seconds"] = perf_counter() - animation_started
    if plot or animate:
        log(f"Display calls: plot {timing['plot_seconds']:.3f} s, "
            f"animation {timing['animation_seconds']:.3f} s "
            "(includes time spent in open windows)")
    return {
        "selection_mode": selection_mode,
        "ball_policy": ball_policy,
        "ball_controls": np.asarray(ball_control_history),
        "ball_path_states": np.asarray(ball_path_states),
        "ball_switch_time": (ball_manager.switch_time
                             if ball_policy == "max_accel_pronav" else None),
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
        "timing": timing,
        "replay": replay,
    }


if __name__ == "__main__":
    num_defenders = 5
    defender_start = (0., 0.)  # Formation center (x, y), in meters.
    defender_spacing = 50.  # Distance between neighboring defenders, in meters.

    # sim(ball_maneuver_duration=2.0, ball_turn_direction=1, verbose=True,
    #     # selection_mode="slowest_symmetric")
    #     selection_mode="median")  # Or "median".

    sim(terminal_heading=np.radians(270), ball_policy="dubins", verbose=True,
            num_defenders=num_defenders, defender_start=defender_start,
            defender_spacing=defender_spacing,
            # selection_mode="slowest_symmetric")
            selection_mode="median")  # Or "median".
