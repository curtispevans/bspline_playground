import numpy as np
import params as P
from dynamics import SimpleDynamics
from intercept import get_cost_metric
from path_manager import PathManager
import matplotlib.pyplot as plt
import time

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

    terminal_heading = np.random.uniform(np.pi/2, 3*np.pi/2)
    target = np.array([0, 0, terminal_heading])

    ball_path_manager = PathManager(R_ball, ball0[:2], ball0[2], target[:2], target[2])

    tf = 100
    t = 0
    ball_hit = False
    hit_target = False

    print("Computing best intercept times and poses for each robot...")
    x1_best_intercept_time, x1_best_intercept_pose, _ = get_cost_metric(
        x1_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    x2_best_intercept_time, x2_best_intercept_pose, _ = get_cost_metric(
        x2_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    x3_best_intercept_time, x3_best_intercept_pose, _ = get_cost_metric(
        x3_robot0, ball0, target,
        v_robot, v_ball, a_max_robot, a_max_ball,
        horizon
    )
    print("Best intercept times and poses computed.")

    x1_path_manager = PathManager(R_robot, x1_robot0[:2], x1_robot0[2], x1_best_intercept_pose[:2], x1_best_intercept_pose[2])
    x2_path_manager = PathManager(R_robot, x2_robot0[:2], x2_robot0[2], x2_best_intercept_pose[:2], x2_best_intercept_pose[2])
    x3_path_manager = PathManager(R_robot, x3_robot0[:2], x3_robot0[2], x3_best_intercept_pose[:2], x3_best_intercept_pose[2])

    print("Starting simulation...")
    x1_robot_poses = []
    x2_robot_poses = []
    x3_robot_poses = []
    ball_poses = []
    while t < tf and not ball_hit and not hit_target:
        ball0 = ball.x
        x1_robot0 = x1_robot.x
        x2_robot0 = x2_robot.x
        x3_robot0 = x3_robot.x

        u_ball = ball_path_manager.update(ball)
        u_x1 = x1_path_manager.update(x1_robot)
        u_x2 = x2_path_manager.update(x2_robot)
        u_x3 = x3_path_manager.update(x3_robot)

        ball.update(u_ball)
        x1_robot.update(u_x1)
        x2_robot.update(u_x2)
        x3_robot.update(u_x3)

        for robot in [x1_robot, x2_robot, x3_robot]:
            if np.linalg.norm(robot.x[:2] - ball.x[:2]) < 10:
                ball_hit = True
                break
        if np.linalg.norm(ball.x[:2] - target[:2]) < 10:
            hit_target = True

        t += P.Ts
        print(f"Time: {t}, Ball Position: {ball.x}, Robots: {x1_robot.x}, {x2_robot.x}, {x3_robot.x}")
        x1_robot_poses.append(x1_robot.x.copy())
        x2_robot_poses.append(x2_robot.x.copy())
        x3_robot_poses.append(x3_robot.x.copy())
        ball_poses.append(ball.x.copy())

    if ball_hit:
        print("Ball hit by a robot!")
    elif hit_target:
        print("Ball reached the target!")
    else:
        print("Simulation ended without the ball being hit.")

    x1_robot_poses = np.array(x1_robot_poses)
    x2_robot_poses = np.array(x2_robot_poses)
    x3_robot_poses = np.array(x3_robot_poses)
    ball_poses = np.array(ball_poses)
    plt.figure(1)
    plt.plot(ball_poses[:,0], ball_poses[:,1], 'ro', label='Ball')
    plt.plot(x1_robot_poses[:,0], x1_robot_poses[:,1], 'bo', label='Robot 1')
    plt.plot(x2_robot_poses[:,0], x2_robot_poses[:,1], 'go', label='Robot 2')
    plt.plot(x3_robot_poses[:,0], x3_robot_poses[:,1], 'mo', label='Robot 3')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sim()