import numpy as np
from scipy.optimize import brentq, minimize_scalar
from path_manager import PathManager, dubins_state_at_time, get_dubins_parameters
import params as P
import matplotlib.pyplot as plt
from dynamics import SimpleDynamics
from pronav import pronav

SELECTION_MODES = ("slowest_symmetric", "median")

def find_intercept(
    x_robot0, x_ball0, x_ball_goal,
    v_robot, v_ball, a_max_robot, a_max_ball,
    horizon, sample_dt=0.1,
):
    R_robot = v_robot**2 / a_max_robot
    R_ball = v_ball**2 / a_max_ball

    def planned_path(T):
        # Dubins paths exist for any pair of poses. Short position separations
        # are not by themselves infeasible, so do not reject them using 2R.
        ball_pose = dubins_state_at_time(x_ball0, x_ball_goal, T, v_ball, a_max_ball)
        if not np.all(np.isfinite(ball_pose)):
            return None

        # This choice makes the interceptor match the ball's heading.
        intercept_pose = ball_pose[:3]
        intercept_pose[2] = ball_pose[2]-np.pi
        params = get_dubins_parameters(
            x_robot0[:2], x_robot0[2],
            intercept_pose[:2], intercept_pose[2],
            R_robot,
        )

        first_arc, second_arc, straight = params[-3:]
        length = R_robot * (first_arc + second_arc) + straight
        if not np.isfinite(length):
            return None
        return intercept_pose, params, length

    def g(T):
        result = planned_path(T)
        if result is None:
            return np.inf
        _, _, length = result
        return v_robot * T - length

    times = np.linspace(
        0.0, horizon, int(np.ceil(horizon / sample_dt)) + 1
    )

    t_left = times[0]
    g_left = g(t_left)
    if g_left == 0:
        pose, params, _ = planned_path(0.0)
        return 0.0, pose, params

    for t_right in times[1:]:
        g_right = g(t_right)

        if g_left < 0 <= g_right:
            t_star = brentq(g, t_left, t_right, xtol=1e-6)
            pose, params, length = planned_path(t_star)
            return t_star, pose, params

        t_left, g_left = t_right, g_right

    return None  # No sign-changing intercept found within this horizon.

def find_pronav_intercept(
    x_robot0, x_ball0, ball_goal,
    v_robot, v_ball, a_max_robot, a_max_ball,
    horizon, hit_radius=5.0, ghost_dt=0.05, sim_dt=P.Ts,
):
    """Find a *fixed* ghost point for proportional-navigation interception.

    Each candidate ghost is a ball position predicted at ``ghost_time``.
    Simulate the robot with PN toward that fixed point (ghost_speed=0), and
    select the candidate yielding the earliest actual robot-ball encounter.
    The ball prediction and the robot are evaluated on the same time axis.

    Returns (hit_time, ghost_pose, ghost_time), or None if none of the sampled
    ghost points gives a hit. The ghost's heading is ignored by PN because its
    speed is zero. Search accuracy depends on ghost_dt and sim_dt.
    """
    if min(v_robot, v_ball, a_max_robot, a_max_ball, sim_dt, ghost_dt) <= 0:
        raise ValueError("Speeds, acceleration limits, and steps must be positive")
    if horizon < 0 or hit_radius < 0:
        raise ValueError("Horizon and hit radius must be nonnegative")

    ball_radius = v_ball**2 / a_max_ball
    ball_path = get_dubins_parameters(
        x_ball0[:2], x_ball0[2], ball_goal[:2], ball_goal[2], ball_radius,
    )
    arc_start, arc_end, straight = ball_path[-3:]
    ball_arrival = (ball_radius * (arc_start + arc_end) + straight) / v_ball
    t_end = min(horizon, ball_arrival)
    times = np.arange(int(np.floor(t_end / sim_dt)) + 1) * sim_dt
    ball_positions = np.array([
        dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball)[:2]
        for t in times
    ])

    def next_state(x, acceleration):
        """Exact constant-control step for SimpleDynamics' planar model."""
        px, py, heading = x
        omega = acceleration / v_robot
        next_heading = heading + omega * sim_dt
        if abs(omega) < 1e-10:
            return np.array([
                px + v_robot * sim_dt * np.cos(heading),
                py + v_robot * sim_dt * np.sin(heading), next_heading,
            ])
        return np.array([
            px + (v_robot / omega) * (np.sin(next_heading) - np.sin(heading)),
            py + (v_robot / omega) * (np.cos(heading) - np.cos(next_heading)),
            next_heading,
        ])

    candidate_times = np.unique(np.r_[np.arange(0, t_end, ghost_dt), t_end])
    best = None
    for ghost_time in candidate_times:
        ghost_pose = dubins_state_at_time(
            x_ball0, ball_goal, ghost_time, v_ball, a_max_ball,
        )
        robot_state = np.asarray(x_robot0, dtype=float).copy()
        if np.linalg.norm(robot_state[:2] - ball_positions[0]) <= hit_radius:
            return 0.0, ghost_pose, float(ghost_time)

        for i in range(1, len(times)):
            if best is not None and times[i] >= best[0]:
                break  # This candidate cannot beat the earliest hit so far.
            acceleration = pronav(
                robot_state, ghost_pose, v_robot, ghost_speed=0.0,
                max_acceleration=a_max_robot,
            )[0]
            robot_state = next_state(robot_state, acceleration)
            if np.linalg.norm(robot_state[:2] - ball_positions[i]) <= hit_radius:
                best = float(times[i]), ghost_pose.copy(), float(ghost_time)
                break

    return best

def find_fastest_intercept(
    x_robot0, x_ball0, x_ball_goal,
    v_robot, v_ball, a_max_robot, a_max_ball,
    horizon, sample_dt=0.1, n_headings=72,
):
    R_robot = v_robot**2 / a_max_robot
    headings = np.linspace(-np.pi, np.pi, n_headings, endpoint=False)
    heading_step = 2 * np.pi / n_headings

    def best_path_at(T):
        ball_state = dubins_state_at_time(x_ball0, x_ball_goal, T, v_ball, a_max_ball)
        ball_xy = ball_state[:2]

        def plan(theta):
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            target = np.array([ball_xy[0], ball_xy[1], theta])

            params = get_dubins_parameters(
                x_robot0[:2], x_robot0[2],
                target[:2], target[2], R_robot,
            )
            first_arc, second_arc, straight = params[-3:]
            length = R_robot * (first_arc + second_arc) + straight
            return length, target, params

        # Sample the whole heading circle, then refine each sampled minimum.
        lengths = np.array([plan(theta)[0] for theta in headings])
        j_best = int(np.argmin(lengths))
        theta_best = headings[j_best]
        length_best = lengths[j_best]

        for j, theta in enumerate(headings):
            if lengths[j] <= lengths[(j - 1) % n_headings] and \
               lengths[j] <= lengths[(j + 1) % n_headings]:
                result = minimize_scalar(
                    lambda th: plan(th)[0],
                    bounds=(theta - heading_step, theta + heading_step),
                    method="bounded",
                )
                if result.fun < length_best:
                    theta_best = result.x
                    length_best = result.fun

        return plan(theta_best)  # length, target pose, path parameters

    def g(T):
        length, _, _ = best_path_at(T)
        return v_robot * T - length

    times = np.linspace(
        0, horizon, int(np.ceil(horizon / sample_dt)) + 1
    )
    t_left = times[0]
    g_left = g(t_left)

    if g_left >= 0:
        length, pose, params = best_path_at(0.0)
        return 0.0, pose, params

    for t_right in times[1:]:
        g_right = g(t_right)
        if g_left < 0 <= g_right:
            t_intercept = brentq(g, t_left, t_right, xtol=1e-6)
            length, pose, params = best_path_at(t_intercept)
            return t_intercept, pose, params

        t_left, g_left = t_right, g_right

    return None


def _symmetric_center_robot_index(xi_robots0, x_ball0, ball_goal,
                                  terminal_thetas, tol=1e-6):
    """Return the centered defender when the heading problem is symmetric."""
    if len(xi_robots0) < 3 or len(terminal_thetas) < 3:
        return None

    theta_center = 0.5 * (terminal_thetas[0] + terminal_thetas[-1])
    if not np.allclose(terminal_thetas + terminal_thetas[::-1],
                       2 * theta_center, atol=tol, rtol=0):
        return None
    center_j = int(np.argmin(np.abs(terminal_thetas - theta_center)))
    if abs(terminal_thetas[center_j] - theta_center) > tol:
        return None

    axis = np.array([np.cos(theta_center), np.sin(theta_center)])
    normal = np.array([-axis[1], axis[0]])
    ball_xy = np.asarray(x_ball0[:2], dtype=float)
    goal_delta = np.asarray(ball_goal[:2], dtype=float) - ball_xy
    if abs(np.dot(goal_delta, normal)) > tol * max(1.0, np.linalg.norm(goal_delta)):
        return None

    poses = [np.asarray(pose, dtype=float) for pose in xi_robots0]
    for center_i, center_pose in enumerate(poses):
        center_delta = center_pose[:2] - ball_xy
        if abs(np.dot(center_delta, normal)) > tol:
            continue
        # A heading on the symmetry axis is unchanged by reflection.
        reflected_center_heading = 2 * theta_center - center_pose[2]
        heading_error = np.arctan2(
            np.sin(reflected_center_heading - center_pose[2]),
            np.cos(reflected_center_heading - center_pose[2]),
        )
        if abs(heading_error) > tol:
            continue

        for first_i in range(len(poses)):
            if first_i == center_i:
                continue
            first_delta = poses[first_i][:2] - ball_xy
            reflected_position = ball_xy + 2 * axis * np.dot(first_delta, axis) - first_delta
            reflected_heading = 2 * theta_center - poses[first_i][2]
            for second_i in range(len(poses)):
                if second_i in (center_i, first_i):
                    continue
                second_pose = poses[second_i]
                heading_error = np.arctan2(
                    np.sin(reflected_heading - second_pose[2]),
                    np.cos(reflected_heading - second_pose[2]),
                )
                if (np.allclose(reflected_position, second_pose[:2],
                                atol=tol, rtol=0) and
                        abs(heading_error) <= tol):
                    midpoint = 0.5 * (poses[first_i][:2] + second_pose[:2])
                    if np.allclose(midpoint, center_pose[:2], atol=tol, rtol=0):
                        return center_i
    return None


def _default_terminal_thetas(ball_heading):
    """Keep the original 10-heading bank and add its exact midpoint."""
    half_span = np.deg2rad(135)
    sampled = np.linspace(-half_span, half_span, 30) + ball_heading
    return np.sort(np.append(sampled, ball_heading))


def _approach_terminal_thetas(x_ball0, ball_goal, radius):
    """Supplement the heading bank with short straight-then-turn endpoints.

    The goal must lie on a radius-R circle tangent to the current heading
    after a nonnegative straight segment. These endpoints keep a narrow short
    branch represented as the intruder approaches its terminal turn, without
    using the intruder's true terminal heading.
    """
    heading = x_ball0[2]
    candidates = list(_default_terminal_thetas(heading))
    forward = np.array([np.cos(heading), np.sin(heading)])
    normal = np.array([-forward[1], forward[0]])
    delta = np.asarray(ball_goal[:2]) - np.asarray(x_ball0[:2])
    along, across = np.dot(delta, forward), np.dot(delta, normal)
    tolerance = 1e-8 * max(radius, np.linalg.norm(delta), 1.)
    for direction in (-1, 1):
        discriminant = radius**2 - (across - direction * radius)**2
        if discriminant < -tolerance * radius:
            continue
        root = (0. if abs(discriminant) <= tolerance * radius
                else np.sqrt(max(0., discriminant)))
        for straight in (along - root, along + root):
            if straight < -tolerance:
                continue
            center = (np.asarray(x_ball0[:2]) + max(0., straight) * forward
                      + direction * radius * normal)
            radial = np.asarray(ball_goal[:2]) - center
            theta = np.arctan2(radial[1], radial[0]) + direction * np.pi / 2
            turn = (direction * (theta - heading)) % (2 * np.pi)
            if turn > np.pi + 1e-8:
                continue
            theta = heading + np.arctan2(np.sin(theta - heading),
                                        np.cos(theta - heading))
            candidates.append(theta)
    candidates = np.sort(candidates)
    return candidates[np.r_[True, np.diff(candidates) > 1e-9]]


def _terminal_turn_candidates(x_ball0, ball_goal, terminal_thetas, radius,
                              goal_radius):
    """Add final arcs ending inside the same goal region as the simulation.

    Project the goal onto each current turn circle. If that circle reaches the
    goal region in at most half a revolution, its endpoint is a valid terminal
    turn. This tolerates tracking error without tightening the turn radius.
    """
    goals = [np.array([ball_goal[0], ball_goal[1], theta], dtype=float)
             for theta in terminal_thetas]
    is_terminal_turn = [False] * len(goals)
    forward = np.array([np.cos(x_ball0[2]), np.sin(x_ball0[2])])
    normal = np.array([-forward[1], forward[0]])
    if goal_radius > 0:
        for direction in (-1, 1):
            center = np.asarray(x_ball0[:2]) + direction * radius * normal
            radial = np.asarray(ball_goal[:2]) - center
            distance = np.linalg.norm(radial)
            if distance == 0 or abs(distance - radius) > goal_radius:
                continue
            theta = np.arctan2(radial[1], radial[0]) + direction * np.pi / 2
            turn = (direction * (theta - x_ball0[2])) % (2 * np.pi)
            if turn > np.pi + 1e-8:
                continue
            theta = x_ball0[2] + np.arctan2(np.sin(theta - x_ball0[2]),
                                          np.cos(theta - x_ball0[2]))
            endpoint = center + radius * radial / distance
            goal = np.r_[endpoint, theta]
            matching = [j for j, candidate in enumerate(goals)
                        if abs(candidate[2] - theta) <= 1e-8]
            if matching:
                goals[matching[0]] = goal
                is_terminal_turn[matching[0]] = True
            else:
                goals.append(goal)
                is_terminal_turn.append(True)
    goals = np.asarray(goals, dtype=float).reshape(-1, 3)
    order = np.argsort(goals[:, 2])
    return goals[order], np.asarray(is_terminal_turn)[order]


def _short_intruder_paths(x_ball0, ball_goal, terminal_thetas, radius,
                          candidate_goals=None):
    """Keep the short length cluster and reject paths requiring looping turns.

    A jump of more than half a turning circumference separates length clusters.
    Also cap total turning at pi radians: when every sampled CSC path is long,
    the shortest of those detours must not become an eligible approach.
    """
    lengths = np.full(len(terminal_thetas), np.inf)
    turns = np.full(len(terminal_thetas), np.inf)
    for j, theta in enumerate(terminal_thetas):
        goal_xy = ball_goal[:2] if candidate_goals is None else candidate_goals[j, :2]
        params = get_dubins_parameters(
            x_ball0[:2], x_ball0[2], goal_xy, theta, radius,
        )
        first_arc, last_arc, straight = params[-3:]
        turns[j] = first_arc + last_arc
        lengths[j] = radius * turns[j] + straight
    ordered_lengths = np.sort(lengths[np.isfinite(lengths)])
    jumps = np.flatnonzero(np.diff(ordered_lengths) > np.pi * radius)
    cutoff = float(ordered_lengths[jumps[0]]) if len(jumps) else np.inf
    eligible = (np.isfinite(lengths) & (lengths <= cutoff + 1e-6)
                & (turns <= np.pi + 1e-8))
    return eligible, lengths, turns, cutoff


def get_cost_metrics(xi_robots0, x_ball0, ball_goal, v_robot, v_ball,
                     a_max_robot, a_max_ball, horizon=100,
                     terminal_thetas=None, *, diagnostics=None, goal_radius=10.0,
                     selection_mode="slowest_symmetric"):
    """Filter long intruder routes, partition headings, and select a target.

    The default bank adds headings for short straight-then-terminal-turn paths.
    Terminal arcs may end within ``goal_radius`` of the goal (10 m by default),
    preserving the turn radius despite small tracking errors. Use zero for an
    exact point goal. A supplied heading bank is filtered without augmentation.
    Only the short length cluster with at most pi radians of total turning is
    evaluated for defender interception. If no short feasible heading belongs
    to a robot, it gets no ghost and the simulation uses direct PN.

    ``selection_mode="slowest_symmetric"`` selects each robot's slowest assigned
    intercept, except that a symmetric centered robot selects the midpoint
    heading. ``"median"`` selects every robot's median assigned intercept time,
    using the lower middle time for an even candidate count. Both modes keep
    the same partition, including ownership of the symmetric midpoint.

    Returns one ``(intercept_time, pose, assigned_intercept_times, theta)``
    tuple per robot. Infeasible robot/heading pairs are ignored. A robot with
    no assigned feasible heading gets ``(None, None, [], None)``.
    If supplied, ``diagnostics`` is populated with the evaluated headings,
    cost matrix, ownership, and selected indices for plotting these decisions.
    """
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    if not np.isfinite(goal_radius) or goal_radius < 0:
        raise ValueError("goal_radius must be finite and nonnegative")
    radius = v_ball**2 / a_max_ball
    if terminal_thetas is None:
        terminal_thetas = _approach_terminal_thetas(x_ball0, ball_goal, radius)
        candidate_goals, terminal_turns = _terminal_turn_candidates(
            x_ball0, ball_goal, terminal_thetas, radius, goal_radius,
        )
        terminal_thetas = candidate_goals[:, 2]
    else:
        candidate_goals = np.array([
            [ball_goal[0], ball_goal[1], theta] for theta in terminal_thetas
        ], dtype=float).reshape(-1, 3)
        terminal_turns = np.zeros(len(candidate_goals), dtype=bool)
    terminal_thetas = np.asarray(terminal_thetas, dtype=float)
    xi_robots0 = list(xi_robots0)
    if not xi_robots0:
        raise ValueError("xi_robots0 must contain at least one robot pose")

    eligible, path_lengths, path_turns, length_cutoff = _short_intruder_paths(
        x_ball0, ball_goal, terminal_thetas, radius, candidate_goals,
    )
    times = np.full((len(xi_robots0), len(terminal_thetas)), np.inf)
    poses = [[None] * len(terminal_thetas) for _ in xi_robots0]
    for j, theta in enumerate(terminal_thetas):
        if not eligible[j]:
            continue
        goal = candidate_goals[j].copy()
        # The intruder forecast ends at its goal; it cannot be intercepted
        # later by treating the clamped endpoint as a stationary intruder.
        candidate_horizon = min(horizon, path_lengths[j] / v_ball)
        for i, xi_robot0 in enumerate(xi_robots0):
            result = find_intercept(
                xi_robot0, x_ball0, goal,
                v_robot, v_ball, a_max_robot, a_max_ball, candidate_horizon,
            )
            if result is not None:
                times[i, j], poses[i][j], _ = result

    # Each heading belongs to the robot that can intercept it first.
    owners = np.argmin(times, axis=0)
    symmetric_center_i = _symmetric_center_robot_index(
        xi_robots0, x_ball0, ball_goal, terminal_thetas,
    )
    center_j = None
    if symmetric_center_i is not None:
        center_j = int(np.argmin(np.abs(terminal_thetas -
                                       0.5 * (terminal_thetas[0] + terminal_thetas[-1]))))
        if np.isfinite(times[symmetric_center_i, center_j]):
            # Keep the midpoint heading in the centered defender's partition.
            owners[center_j] = symmetric_center_i

    results = []
    selected_indices = np.full(len(xi_robots0), -1, dtype=int)
    selection_reasons = ["unassigned"] * len(xi_robots0)
    for i in range(len(xi_robots0)):
        assigned = [j for j, owner in enumerate(owners)
                    if owner == i and np.isfinite(times[i, j])]
        if not assigned:
            results.append((None, None, [], None))
            continue
        if selection_mode == "slowest_symmetric":
            if i == symmetric_center_i and center_j in assigned:
                selected_j = center_j
                selection_reasons[i] = "symmetry"
            else:
                selected_j = max(assigned, key=lambda j: times[i, j])
                selection_reasons[i] = "slowest"
        else:
            # Select an actual sample, using the lower median for even counts.
            ordered = sorted(assigned, key=lambda j: times[i, j])
            selected_j = ordered[(len(ordered) - 1) // 2]
            selection_reasons[i] = "median"
        selected_indices[i] = selected_j
        results.append((float(times[i, selected_j]), poses[i][selected_j],
                        [float(times[i, j]) for j in assigned],
                        float(terminal_thetas[selected_j])))
    if diagnostics is not None:
        diagnostics.update(
            selection_mode=selection_mode,
            selection_reasons=selection_reasons,
            terminal_thetas=terminal_thetas.copy(),
            candidate_goals=candidate_goals.copy(),
            terminal_turn_mask=terminal_turns,
            goal_radius=float(goal_radius),
            eligible_mask=eligible.copy(),
            intruder_path_lengths=path_lengths,
            intruder_path_turns=path_turns,
            short_path_length_cutoff=length_cutoff,
            intercept_times=times.copy(),
            owners=np.where(np.isfinite(times).any(axis=0), owners, -1),
            selected_indices=selected_indices,
            selected_poses=np.array([
                np.full(3, np.nan) if result[1] is None else result[1].copy()
                for result in results
            ]),
            robot_poses=np.array(xi_robots0, dtype=float).copy(),
            ball_pose=np.asarray(x_ball0, dtype=float).copy(),
            goal=np.asarray(ball_goal, dtype=float).copy(),
            v_ball=float(v_ball), a_max_ball=float(a_max_ball),
            horizon=float(horizon),
        )
    return results


def get_cost_metric(xi_robot0, x_ball0, ball_goal, v_robot, v_ball,
                    a_max_robot, a_max_ball, horizon=100,
                    team_robot_poses=None, robot_index=None, goal_radius=10.0,
                    selection_mode="slowest_symmetric"):
    """Return a robot's selected assigned intercept and its terminal heading.

    When ``team_robot_poses`` is provided, headings are assigned to the team
    member with the earliest intercept. ``selection_mode`` chooses the slowest
    intercept with a symmetric midpoint override, or the median time. Without
    team poses, the same rule selects among this robot's feasible headings.
    ``goal_radius`` matches the goal-region tolerance used by the team planner.
    Results are ``(intercept_time, pose, intercept_times, terminal_theta)``.
    """
    if team_robot_poses is not None:
        if robot_index is None or not 0 <= robot_index < len(team_robot_poses):
            raise ValueError("robot_index must identify a robot in team_robot_poses")
        return get_cost_metrics(
            team_robot_poses, x_ball0, ball_goal, v_robot, v_ball,
            a_max_robot, a_max_ball, horizon, goal_radius=goal_radius,
            selection_mode=selection_mode,
        )[robot_index]

    # A single robot owns every eligible feasible heading; use the same bank.
    results = get_cost_metrics(
        [xi_robot0], x_ball0, ball_goal, v_robot, v_ball,
        a_max_robot, a_max_ball, horizon, goal_radius=goal_radius,
        selection_mode=selection_mode,
    )
    return results[0]

def get_cost_metric_pronav(
    xi_robot0, x_ball0, ball_goal, v_robot, v_ball,
    a_max_robot, a_max_ball, horizon=100, hit_radius=10.0,
    ghost_dt=0.05,
):
    """Return (predicted_hit_time, fixed_ghost_pose, ball_time_at_ghost).

    Evaluate PN against the ball's *actual assigned trajectory* using its
    given terminal heading. Do not change the terminal heading during the
    cost calculation unless the ball will really fly a different trajectory.
    """
    result = find_pronav_intercept(
        xi_robot0, x_ball0, ball_goal,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon, hit_radius=hit_radius, ghost_dt=ghost_dt,
    )
    return result if result is not None else (None, None, None)

def test(selection_mode="slowest_symmetric"):
    """Inspect the same heading bank and defender formation as the simulation."""
    from assignment_plotting import plot_assignment

    robot_poses = [np.array([0., 100., 0.]),
                   np.array([0., 0., 0.]),
                   np.array([0., -100., 0.])]
    ball_pose = np.array([1000., 0., np.pi])
    ball_goal = np.array([0., 0., np.pi])
    diagnostics = {}
    results = get_cost_metrics(
        robot_poses, ball_pose, ball_goal, 30., 100.,
        2 * P.gravity, P.max_acceleration, diagnostics=diagnostics,
        selection_mode=selection_mode,
    )
    for i, result in enumerate(results, 1):
        if result[3] is None:
            print(f"Defender {i}: no assigned feasible heading")
        else:
            print(f"Defender {i}: {len(result[2])} assigned paths, "
                  f"{diagnostics['selection_reasons'][i - 1]} terminal heading "
                  f"{np.rad2deg(result[3]):.1f} deg, "
                  f"intercept time {result[0]:.2f} s")
    return plot_assignment(diagnostics)


def test_as_time_moves():
    x_robot_initial = np.array([0, 100, 0])
    x_ball_initial = np.array([1000, 0, np.pi])
    ball_goal = np.array([0, 0, np.pi/2])
    v_robot = 30
    v_ball = 100
    a_max_robot = 2*P.gravity
    a_max_ball = P.max_acceleration
    horizon = 100
    R_ball = v_ball**2 / a_max_ball
    x_robot0 = x_robot_initial
    x_ball0 = x_ball_initial
    robot = SimpleDynamics(x_robot0, v_robot, a_max_robot)
    ball = SimpleDynamics(x_ball0, v_ball, a_max_ball)
    ball_path_manager = PathManager(R_ball, x_ball0[:2], x_ball0[2], ball_goal[:2], ball_goal[2])

    ball_start = x_ball0[:2].copy()
    ball_end = ball_goal[:2].copy()
    ball_hit = False
    times = np.linspace(0, horizon, 10)
    t = 0
    t_check = 1
    t_add = 1
    x_ball_t = dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball)
    result = find_intercept(
        x_robot0, x_ball_t, ball_goal,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon - t,
    )
    t_star, pose, params = result
    while t < horizon and not ball_hit:
        x_robot0 = robot.x
        x_ball0 = ball.x
        if t_check - P.Ts <= t <= t_check + P.Ts:
            x_ball_t = dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball)
            # result = find_intercept(
            #     x_robot0, x_ball_t, ball_goal,
            #     v_robot, v_ball, a_max_robot, a_max_ball,
            #     horizon - t,
            # )
            # if result is None:
            #     continue
            # t_star, pose, params = result
            robot_path = np.array([dubins_state_at_time(x_robot0, pose, t, v_robot, a_max_robot) for t in np.linspace(0, t_star, 100)])
            ball_path = np.array([dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball) for t in np.linspace(0, t_star, 100)])
            plt.plot(robot_path[:, 0], robot_path[:, 1], 'r-', label='Intercept Path')
            plt.plot(ball_path[:, 0], ball_path[:, 1], 'b-', label='Ball Path')
            plt.plot(ball_start[0], ball_start[1], 'go', label='Ball Start')
            plt.plot(x_robot0[0], x_robot0[1], 'bo', label='Robot Start')
            plt.plot(x_ball_t[0], x_ball_t[1], 'ko', label='Ball Pose')
            plt.plot(ball_goal[0], ball_goal[1], 'mo', label='Ball Goal')
            plt.legend()
            plt.axis('equal')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Intercept Path at t={t:.2f}")
            plt.show()
            t_check += t_add
        ghost = dubins_state_at_time(x_robot_initial, pose, min(t,t_star), v_robot, a_max_robot)
        u_robot = pronav(robot.x, ghost, v_robot, ghost_speed=v_ball, navigation_constant=3, max_acceleration=a_max_robot)
        u_ball = ball_path_manager.update(ball)
        robot.update(u_robot)
        ball.update(u_ball)
        t += P.Ts
        if np.linalg.norm(robot.x[:2] - ball.x[:2]) < 5.0:
            print(f"Robot intercepted the ball at time {t:.2f} seconds.")
            ball_hit = True
        if np.linalg.norm(ball.x[:2] - ball_goal[:2]) < 5.0:
            print(f"Ball reached the goal at time {t:.2f} seconds.")
            ball_hit = True

def test_as_time_moves_1(controller="assigned_ghost", plot_times=(0, 3, 6, 8)):
    """Compare the simulated motion against one fixed, time-indexed plan.

    Use controller="assigned_ghost" for PN to a fixed point selected from the
    predicted ball path, or "path_manager" to validate the Dubins path itself.
    "moving_ghost" uses PN to the ball's current pose and velocity.
    Use controller="path_ghost" to reproduce the earlier path-tracking test,
    where proportional navigation pursues a reference on the robot's path.
    """
    x_robot_initial = np.array([0., 100., 0.])
    x_ball_initial = np.array([1000., 0., np.pi])
    ball_goal = np.array([0., 0., np.pi / 2])
    v_robot, v_ball = 30., 100.
    a_max_robot, a_max_ball = 2 * P.gravity, P.max_acceleration
    horizon = 100.

    result = find_intercept(
        x_robot_initial, x_ball_initial, ball_goal,
        v_robot, v_ball, a_max_robot, a_max_ball, horizon,
    )
    if result is None:
        raise RuntimeError("No planned intercept within the horizon")
    t_star, pose, params = result
    print(f"Planned intercept: t={t_star:.3f}, position={pose[:2]}")

    robot = SimpleDynamics(x_robot_initial.copy(), v_robot, a_max_robot)
    ball = SimpleDynamics(x_ball_initial.copy(), v_ball, a_max_ball)
    robot_path_manager = PathManager(
        v_robot**2 / a_max_robot, x_robot_initial[:2],
        x_robot_initial[2], pose[:2], pose[2],
    )
    ball_path_manager = PathManager(
        v_ball**2 / a_max_ball, x_ball_initial[:2],
        x_ball_initial[2], ball_goal[:2], ball_goal[2],
    )
    if controller not in ("assigned_ghost", "path_manager", "moving_ghost", "path_ghost"):
        raise ValueError(
            "controller must be 'assigned_ghost', 'path_manager', "
            "'moving_ghost', or 'path_ghost'"
        )

    assigned_ghost = None
    if controller == "assigned_ghost":
        assignment = find_pronav_intercept(
            x_robot_initial, x_ball_initial, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball, horizon,
        )
        if assignment is None:
            print("No sampled stationary ghost produced an intercept.")
            return
        predicted_hit, assigned_ghost, ghost_time = assignment
        print(f"Assigned PN ghost: {assigned_ghost[:2]} "
              f"(ball position predicted at t={ghost_time:.2f} s; "
              f"PN hit predicted at t={predicted_hit:.2f} s)")

    plan_times = np.linspace(0, t_star, 300)
    robot_plan = np.array([
        dubins_state_at_time(x_robot_initial, pose, tau, v_robot, a_max_robot)
        for tau in plan_times
    ])
    ball_plan = np.array([
        dubins_state_at_time(x_ball_initial, ball_goal, tau, v_ball, a_max_ball)
        for tau in plan_times
    ])
    robot_history = [robot.x.copy()]
    ball_history = [ball.x.copy()]
    closest_separation = np.inf
    closest_time = 0.0

    for step in range(int(horizon / P.Ts) + 1):
        t = step * P.Ts
        if any(abs(t - mark) <= P.Ts / 2 for mark in plot_times):
            robot_trace = np.array(robot_history)
            ball_trace = np.array(ball_history)
            fig, ax = plt.subplots()
            ax.plot(robot_plan[:, 0], robot_plan[:, 1], 'r-', label='Robot plan')
            ax.plot(ball_plan[:, 0], ball_plan[:, 1], 'b-', label='Ball plan')
            ax.plot(robot_trace[:, 0], robot_trace[:, 1], 'r:', label='Robot actual')
            ax.plot(ball_trace[:, 0], ball_trace[:, 1], 'b:', label='Ball actual')
            ax.plot(*robot.x[:2], 'ro', label='Robot now')
            ax.plot(*ball.x[:2], 'bo', label='Ball now')
            ax.plot(*pose[:2], 'kx', label='Planned intercept')
            if assigned_ghost is not None:
                ax.plot(*assigned_ghost[:2], 'm*', markersize=10,
                        label='Assigned stationary ghost')
            ax.set(xlabel='X', ylabel='Y', title=f'{controller} at t={t:.2f} s')
            ax.axis('equal')
            ax.legend()
            plt.show()

        if controller == "path_manager":
            u_robot = robot_path_manager.update(robot)
        elif controller == "assigned_ghost":
            u_robot = pronav(
                robot.x, assigned_ghost, robot_speed=v_robot, ghost_speed=0.0,
                navigation_constant=3, max_acceleration=a_max_robot,
            )
        elif controller == "moving_ghost":
            # Actual ball is the moving target. The first speed is the
            # robot's speed; ghost_speed is the ball's speed.
            u_robot = pronav(
                robot.x, ball.x, robot_speed=v_robot, ghost_speed=v_ball,
                navigation_constant=3, max_acceleration=a_max_robot,
            )
        else:  # controller == "path_ghost": path-reference experiment
            ghost = dubins_state_at_time(
                x_robot_initial, pose, min(t, t_star), v_robot, a_max_robot,
            )
            u_robot = pronav(
                robot.x, ghost, v_robot, ghost_speed=v_robot,
                navigation_constant=3, max_acceleration=a_max_robot,
            )
        u_ball = ball_path_manager.update(ball)
        robot.update(u_robot)
        ball.update(u_ball)
        robot_history.append(robot.x.copy())
        ball_history.append(ball.x.copy())

        separation = np.linalg.norm(robot.x[:2] - ball.x[:2])
        if separation < closest_separation:
            closest_separation = separation
            closest_time = t + P.Ts
        if separation < 2.5:
            print(f"Robot intercepted the ball at t={t + P.Ts:.2f} s "
                  f"(separation={separation:.2f}).")
            return
        if np.linalg.norm(ball.x[:2] - ball_goal[:2]) < 5.0:
            print(f"Ball reached the goal at t={t + P.Ts:.2f} s "
                  f"(separation={separation:.2f}).")
            print(f"Closest approach: {closest_separation:.2f} "
                  f"at t={closest_time:.2f} s.")
            return
    

if __name__ == "__main__":
    test(selection_mode="slowest_symmetric")  # Or "median".
    # test_as_time_moves_1(controller="assigned_ghost")
