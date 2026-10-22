import numpy as np
import params as P
from dynamics import SimpleDynamics
from intercept import get_cost_metric
from path_manager import PathManager
from pronav import pronav
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from plotting_animation import plot_simulation, animate_simulation


def _pose_for_history(pose):
    """Store an unavailable intercept pose as NaNs for animation playback."""
    if pose is None:
        return np.full(3, np.nan)
    return np.asarray(pose, dtype=float).copy()



def sim():
    ball0 = np.array([1000, 0, np.pi])
    x1_robot0 = np.array([0, 100, 0])
    x2_robot0 = np.array([0, 0, 0])
    x3_robot0 = np.array([0, -100, 0])

    v_robot = 30
    v_ball = 100
    a_max_robot = 2*P.gravity
    a_max_ball = P.max_acceleration
    horizon = 100
    R_ball = v_ball**2/a_max_ball
    R_robot = v_robot**2/a_max_robot

    x1_robot = SimpleDynamics(x1_robot0, v_robot, a_max_robot)
    x2_robot = SimpleDynamics(x2_robot0, v_robot, a_max_robot)
    x3_robot = SimpleDynamics(x3_robot0, v_robot, a_max_robot)
    ball = SimpleDynamics(ball0, v_ball, a_max_ball)

    terminal_heading = 0*4.422615269729036 + np.pi/2 + 0*np.random.uniform(np.pi/2, 3*np.pi/2)
    print(f"terminal_heading: {terminal_heading}")
    target = np.array([0, 0, terminal_heading])

    ball_path_manager = PathManager(R_ball, ball0[:2], ball0[2], target[:2], target[2])

    tf = 100
    t = 0
    ball_hit = False
    hit_target = False

    print("Computing best intercept times and poses for each robot...")
    x1_best_intercept_time, x1_best_intercept_pose, _, x1_terminal_theta = get_cost_metric(
        x1_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    x2_best_intercept_time, x2_best_intercept_pose, _, x2_terminal_theta = get_cost_metric(
        x2_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    x3_best_intercept_time, x3_best_intercept_pose, _, x3_terminal_theta = get_cost_metric(
        x3_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    # The robot path managers are not used below. Avoid constructing them here:
    # an unavailable intercept pose is None and has no [:2] slice.

    print("Starting simulation...")
    x1_robot_poses = [x1_robot.x.copy()]
    x2_robot_poses = [x2_robot.x.copy()]
    x3_robot_poses = [x3_robot.x.copy()]
    ball_poses = [ball.x.copy()]
    sim_times = [t]
    x1_ghost_poses = [_pose_for_history(x1_best_intercept_pose)]
    x2_ghost_poses = [_pose_for_history(x2_best_intercept_pose)]
    x3_ghost_poses = [_pose_for_history(x3_best_intercept_pose)]
    x1_controls = [np.nan]
    x2_controls = [np.nan]
    x3_controls = [np.nan]
    t_check = 0.5
    t_check_step = 0.5
    while t < tf and not ball_hit and not hit_target:
        ball0 = ball.x
        x1_robot0 = x1_robot.x
        x2_robot0 = x2_robot.x
        x3_robot0 = x3_robot.x

        # compute the path manager every second for each robot
        if t_check - P.Ts < t < t_check + P.Ts:
            print("Computing best intercept times and poses for each robot...")
            x1_best_intercept_time, x1_best_intercept_pose, _, x1_terminal_theta = get_cost_metric(
                            x1_robot0, ball0, target,
                            v_robot, v_ball, a_max_robot, a_max_ball,
                            horizon
                        )
            x2_best_intercept_time, x2_best_intercept_pose, _, x2_terminal_theta = get_cost_metric(
                x2_robot0, ball0, target,
                v_robot, v_ball, a_max_robot, a_max_ball,
                horizon
            )
            x3_best_intercept_time, x3_best_intercept_pose, _, x3_terminal_theta = get_cost_metric(
                x3_robot0, ball0, target,
                v_robot, v_ball, a_max_robot, a_max_ball,
                horizon
            )

            if x1_best_intercept_pose is None:
                print("No valid intercept found for Robot 1.")
                print(f"robot 1 position: {x1_robot.x}")
                print(f"ball position: {ball0}")
                # pronav to the ball
                # x1_control = pronav(x1_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
                
            if x2_best_intercept_pose is None:
                print("No valid intercept found for Robot 2.")
                print(f"robot 2 position: {x2_robot.x}")
                print(f"ball position: {ball0}")
                # pronav to the ball
                # x2_control = pronav(x2_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
                
            if x3_best_intercept_pose is None:
                print("No valid intercept found for Robot 3.")
                print(f"robot 3 position: {x3_robot.x}")
                print(f"ball position: {ball0}")
                # pronav to the ball
                # x3_control = pronav(x3_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)

            t_check += t_check_step

        # if np.allclose(x1_best_intercept_pose, target, atol=1e-2):
        #     x1_control = pronav(x1_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
        # if np.allclose(x2_best_intercept_pose, target, atol=1e-2):
        #     x2_control = pronav(x2_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
        # if np.allclose(x3_best_intercept_pose, target, atol=1e-2):
        #     x3_control = pronav(x3_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)

        else:
            if np.linalg.norm(x1_robot0[:2] - ball0[:2]) < R_ball:
                print("Defender 1 is close to the ball.")
                x1_best_intercept_pose = ball0
            if np.linalg.norm(x2_robot0[:2] - ball0[:2]) < R_ball:
                print("Defender 2 is close to the ball.")
                x2_best_intercept_pose = ball0
            if np.linalg.norm(x3_robot0[:2] - ball0[:2]) < R_ball:
                print("Defender 3 is close to the ball.")
                x3_best_intercept_pose = ball0

            if np.linalg.norm(x1_best_intercept_pose[:2] - target[:2]) < R_robot:
                print("Defender 1's best intercept pose is at the target.")
                x1_best_intercept_pose = ball0
            if np.linalg.norm(x2_best_intercept_pose[:2] - target[:2]) < R_robot:
                print("Defender 2's best intercept pose is at the target.")
                x2_best_intercept_pose = ball0
            if np.linalg.norm(x3_best_intercept_pose[:2] - target[:2]) < R_robot:
                print("Defender 3's best intercept pose is at the target.")
                x3_best_intercept_pose = ball0

            
            x1_control = pronav(x1_robot.x, x1_best_intercept_pose, v_robot, ghost_speed=0, navigation_constant=3, max_acceleration=a_max_robot)
            x2_control = pronav(x2_robot.x, x2_best_intercept_pose, v_robot, ghost_speed=0, navigation_constant=3, max_acceleration=a_max_robot)
            x3_control = pronav(x3_robot.x, x3_best_intercept_pose, v_robot, ghost_speed=0, navigation_constant=3, max_acceleration=a_max_robot)


        u_ball = ball_path_manager.update(ball)
        u_x1 = x1_control
        u_x2 = x2_control
        u_x3 = x3_control

        ball.update(u_ball)
        x1_robot.update(u_x1)
        x2_robot.update(u_x2)
        x3_robot.update(u_x3)

        for robot in [x1_robot, x2_robot, x3_robot]:
            if np.linalg.norm(robot.x[:2] - ball.x[:2]) < 10:
                ball_hit = True
                break
        if np.linalg.norm(ball.x[:2] - target[:2]) < 5:
            print('here')
            print(f'norm value {np.linalg.norm(ball.x[:2] - target[:2])}')
            hit_target = True

        t += P.Ts
        sim_times.append(t)
        x1_robot_poses.append(x1_robot.x.copy())
        x2_robot_poses.append(x2_robot.x.copy())
        x3_robot_poses.append(x3_robot.x.copy())
        ball_poses.append(ball.x.copy())
        x1_ghost_poses.append(_pose_for_history(x1_best_intercept_pose))
        x2_ghost_poses.append(_pose_for_history(x2_best_intercept_pose))
        x3_ghost_poses.append(_pose_for_history(x3_best_intercept_pose))
        x1_controls.append(float(np.asarray(u_x1).reshape(-1)[0]))
        x2_controls.append(float(np.asarray(u_x2).reshape(-1)[0]))
        x3_controls.append(float(np.asarray(u_x3).reshape(-1)[0]))

    if ball_hit:
        print("Ball hit by a robot!")
    elif hit_target:
        print("Ball reached the target!")
    else:
        print("Simulation ended without the ball being hit.")

    robot_poses = [
        np.asarray(x1_robot_poses),
        np.asarray(x2_robot_poses),
        np.asarray(x3_robot_poses),
    ]
    ghost_poses = [
        np.asarray(x1_ghost_poses),
        np.asarray(x2_ghost_poses),
        np.asarray(x3_ghost_poses),
    ]
    animate_simulation(
        np.asarray(sim_times),
        robot_poses,
        np.asarray(ball_poses),
        ghost_poses,
        [np.asarray(x1_controls), np.asarray(x2_controls), np.asarray(x3_controls)],
        target,
    )


if __name__ == "__main__":
    sim()
