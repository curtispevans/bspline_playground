import numpy as np
from scipy.optimize import brentq, minimize_scalar
from path_manager import dubins_state_at_time, get_dubins_parameters
import params as P
import matplotlib.pyplot as plt

def find_intercept(
    x_robot0, x_ball0, x_ball_goal,
    v_robot, v_ball, a_max_robot, a_max_ball,
    horizon, sample_dt=0.1,
):
    R_robot = v_robot**2 / a_max_robot

    def planned_path(T):
        ball_pose = dubins_state_at_time(x_ball0, x_ball_goal, T, v_ball, a_max_ball)

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
        return intercept_pose, params, length

    def g(T):
        _, _, length = planned_path(T)
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


def test():
    x_robot0 = np.array([0, 100, 0])
    x_ball0 = np.array([1000, 0, np.pi])
    ball_goal = np.array([0, 0, np.pi/2])

    v_robot = 30
    v_ball = 100
    a_max_robot = 2*P.gravity
    a_max_ball = P.max_acceleration
    horizon = 100

    t_star, pose, params = find_fastest_intercept(
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
    terminal_thetas = np.linspace(-np.pi/2, np.pi/2, 25) + np.pi
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
    
        t1_star, pose, params = find_fastest_intercept(
            x1_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        t2_star, pose, params = find_fastest_intercept(
            x2_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        intercept2_times.append(t2_star)

        t3_star, pose, params = find_fastest_intercept(
            x3_robot0, x_ball0, ball_goal,
            v_robot, v_ball, a_max_robot, a_max_ball,
            horizon,
        )
        intercept3_times.append(t3_star)
        intercept1_times.append(t1_star)

    terminal_thetas = np.rad2deg(np.array(terminal_thetas))
    plt.plot(terminal_thetas, intercept1_times, 'o-', label='Robot 1')
    plt.plot(terminal_thetas, intercept2_times, 'o-', label='Robot 2')
    plt.plot(terminal_thetas, intercept3_times, 'o-', label='Robot 3')
    plt.legend()
    plt.xlabel("Terminal Theta")
    plt.ylabel("Intercept Time")
    plt.title("Intercept Time vs Terminal Theta")
    plt.show()

if __name__ == "__main__":
    test()