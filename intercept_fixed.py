import numpy as np
from scipy.optimize import brentq, minimize_scalar
from path_manager import PathManager, dubins_state_at_time, get_dubins_parameters
import params as P
import matplotlib.pyplot as plt
from dynamics import SimpleDynamics
from pronav import pronav

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

        # A head-on intercept: the robot heading is opposite the ball heading.
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
    ball_poses = np.array([
        dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball)
        for t in times
    ])
    return find_pronav_intercept_on_trajectory(
        x_robot0, ball_poses, v_robot, a_max_robot,
        hit_radius=hit_radius, ghost_dt=ghost_dt, sim_dt=sim_dt,
    )


def find_pronav_intercept_on_trajectory(
    x_robot0, ball_poses, v_robot, a_max_robot,
    hit_radius=5.0, ghost_dt=0.05, sim_dt=P.Ts,
):
    """Choose a stationary ghost from any time-sampled ball forecast.

    ``ball_poses[i]`` is the forecast ball pose at time ``i * sim_dt``
    relative to the latest observation. Returns (hit_time, ghost_pose,
    ghost_time), all times relative to that observation, or None.
    """
    ball_poses = np.asarray(ball_poses, dtype=float)
    if ball_poses.ndim != 2 or ball_poses.shape[1] != 3 or len(ball_poses) == 0:
        raise ValueError("ball_poses must have shape (n, 3) with n >= 1")
    if not np.all(np.isfinite(ball_poses)):
        raise ValueError("ball_poses must be finite")
    if min(v_robot, a_max_robot, ghost_dt, sim_dt) <= 0 or hit_radius < 0:
        raise ValueError("Speeds, limits, and steps must be positive")

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

    stride = max(1, int(round(ghost_dt / sim_dt)))
    indices = np.unique(np.r_[np.arange(0, len(ball_poses), stride),
                               len(ball_poses) - 1])
    best = None
    for ghost_index in indices:
        ghost_time = ghost_index * sim_dt
        ghost_pose = ball_poses[ghost_index]
        robot_state = np.asarray(x_robot0, dtype=float).copy()
        if np.linalg.norm(robot_state[:2] - ball_poses[0, :2]) <= hit_radius:
            return 0.0, ghost_pose.copy(), float(ghost_time)

        for i in range(1, len(ball_poses)):
            t = i * sim_dt
            if best is not None and t >= best[0]:
                break  # This candidate cannot beat the earliest hit so far.
            acceleration = pronav(
                robot_state, ghost_pose, v_robot, ghost_speed=0.0,
                max_acceleration=a_max_robot,
            )[0]
            robot_state = next_state(robot_state, acceleration)
            if np.linalg.norm(robot_state[:2] - ball_poses[i, :2]) <= hit_radius:
                best = float(t), ghost_pose.copy(), float(ghost_time)
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

def get_cost_metric(
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

def test():
    x_robot0 = np.array([0, 100, 0])
    x_ball0 = np.array([1000, 0, np.pi])
    ball_goal = np.array([0, 0, np.pi/2])
    
    v_robot = 30
    v_ball = 100
    a_max_robot = 2*P.gravity
    a_max_ball = P.max_acceleration
    horizon = 100

    t_star, pose, params = find_intercept(
        x_robot0, x_ball0, ball_goal,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon,
    )
    robot_path = np.array([dubins_state_at_time(x_robot0, pose, t, v_robot, a_max_robot) for t in np.linspace(0, t_star, 100)])
    ball_path = np.array([dubins_state_at_time(x_ball0, ball_goal, t, v_ball, a_max_ball) for t in np.linspace(0, t_star, 100)])
    print("Intercept time:", t_star)
    print("Intercept pose:", pose)
    

    # Plot the intercept path
    plt.plot(robot_path[:, 0], robot_path[:, 1], 'r-', label='Intercept Path')
    plt.plot(ball_path[:, 0], ball_path[:, 1], 'b-', label='Ball Path')
    plt.plot(x_robot0[0], x_robot0[1], 'bo', label='Robot Start')
    plt.plot(x_ball0[0], x_ball0[1], 'go', label='Ball Start')
    plt.plot(ball_goal[0], ball_goal[1], 'mo', label='Ball Goal')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Intercept Path")
    plt.show()

    print('Running for different terminal headings...')
    terminal_thetas = np.linspace(-np.pi/2, np.pi/2, 10) + np.pi
    intercept1_times = []
    intercept2_times = []
    intercept3_times = []
    for theta in terminal_thetas:
        x1_robot0 = np.array([0, 100, 0])
        x2_robot0 = np.array([0, -100, 0])
        x3_robot0 = np.array([0,0,0])
        x_ball0 = np.array([1000, 0, np.pi])
        ball_goal = np.array([0, 0, theta])
    
        v_robot = 30
        v_ball = 100
        a_max_robot = 2*P.gravity
        a_max_ball = P.max_acceleration
        horizon = 100
    
        t1_star, pose, params = find_intercept(
            x1_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        t2_star, pose, params = find_intercept(
            x2_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        intercept2_times.append(t2_star)

        t3_star, pose, params = find_intercept(
            x3_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        intercept3_times.append(t3_star)
        intercept1_times.append(t1_star)

    robot1_assigned_terminal_theta = np.rad2deg(terminal_thetas[np.argmin(intercept1_times)])
    robot2_assigned_terminal_theta = np.rad2deg(terminal_thetas[np.argmin(intercept2_times)])
    robot3_assigned_terminal_theta = np.rad2deg(terminal_thetas[np.argmin(intercept3_times)])
    print(f'Robot 1 assignment terminal heading for ball {robot1_assigned_terminal_theta}')
    print(f'Robot 2 assignment terminal heading for ball {robot2_assigned_terminal_theta}')
    print(f'Robot 3 assignment terminal heading for ball {robot3_assigned_terminal_theta}')
    terminal_thetas = np.rad2deg(np.array(terminal_thetas))
    plt.plot(terminal_thetas, intercept1_times, 'o-', label='Robot 1')
    plt.plot(terminal_thetas, intercept2_times, 'o-', label='Robot 2')
    plt.plot(terminal_thetas, intercept3_times, 'o-', label='Robot 3')
    plt.legend()
    plt.xlabel("Terminal Theta")
    plt.ylabel("Intercept Time")
    plt.title("Intercept Time vs Terminal Theta")
    plt.show()

def test_as_time_moves(controller="assigned_ghost", plot_times=(0, 3, 6, 8)):
    """Compare the simulated motion against one fixed, time-indexed plan.

    Use controller="assigned_ghost" for PN to a fixed point selected from the
    predicted ball path, or "path_manager" to validate the Dubins path itself.
    "moving_ghost" uses PN to the ball's current pose and velocity.
    Use controller="path_ghost" to reproduce the earlier path-tracking test,
    where proportional navigation pursues a reference on the robot's path.
    """
    x_robot_initial = np.array([0., -100., 0.])
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
        if separation < 5.0:
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
    # test()
    test_as_time_moves(controller="assigned_ghost")
