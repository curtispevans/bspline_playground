import numpy as np
import params as P
from dynamics import SimpleDynamics
from intercept import get_cost_metric
from path_manager import PathManager
from pronav_controller import pronav
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


def _pose_for_history(pose):
    """Store an unavailable intercept pose as NaNs for animation playback."""
    if pose is None:
        return np.full(3, np.nan)
    return np.asarray(pose, dtype=float).copy()


def animate_simulation(times, robot_poses, ball_poses, ghost_poses, controls, target):
    """Show a replayable simulation with a frame slider and step controls.

    Use the slider or the left/right arrow keys to inspect individual frames.
    Space toggles playback. The faint lines show each complete trajectory;
    bright lines show the trail up to the selected frame.
    """
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("Robot 1", "Robot 2", "Robot 3")
    n_frames = len(times)
    if n_frames == 0:
        print("No simulation frames were recorded.")
        return None, None

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.subplots_adjust(bottom=0.24)

    all_xy = [ball_poses[:, :2], *(poses[:, :2] for poses in robot_poses), target[None, :2]]
    all_xy = np.vstack(all_xy)
    span = np.ptp(all_xy, axis=0)
    margin = max(0.08 * float(np.max(span)), 20.0)
    ax.set_xlim(np.min(all_xy[:, 0]) - margin, np.max(all_xy[:, 0]) + margin)
    ax.set_ylim(np.min(all_xy[:, 1]) - margin, np.max(all_xy[:, 1]) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Robot and Ball Simulation — replay controls below")
    ax.grid(True, alpha=0.25)
    ax.plot(target[0], target[1], marker="*", color="black", markersize=14, label="Ball goal")

    trail_lines = []
    robot_markers = []
    heading_lines = []
    ghost_markers = []
    for poses, color, label in zip(robot_poses, colors, labels):
        ax.plot(poses[:, 0], poses[:, 1], color=color, alpha=0.18, linewidth=1.5)
        trail, = ax.plot([], [], color=color, linewidth=2.0, label=f"{label} trail")
        marker, = ax.plot([], [], "o", color=color, markersize=8)
        heading, = ax.plot([], [], color=color, linewidth=2.5)
        ghost, = ax.plot([], [], marker="D", linestyle="None", color=color,
                         markerfacecolor="none", markersize=9,
                         label=f"{label} computed intercept")
        trail_lines.append(trail)
        robot_markers.append(marker)
        heading_lines.append(heading)
        ghost_markers.append(ghost)

    ax.plot(ball_poses[:, 0], ball_poses[:, 1], color="magenta", alpha=0.18, linewidth=1.5)
    ball_trail, = ax.plot([], [], color="magenta", linewidth=2.0, label="Ball trail")
    ball_marker, = ax.plot([], [], "o", color="magenta", markersize=9, label="Ball")
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                     family="monospace", fontsize=9,
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    ax.legend(loc="best", fontsize=8)

    slider_ax = fig.add_axes([0.15, 0.12, 0.70, 0.035])
    frame_slider = Slider(slider_ax, "Frame", 0, n_frames - 1,
                          valinit=0, valstep=1)
    back_ax = fig.add_axes([0.27, 0.035, 0.12, 0.055])
    play_ax = fig.add_axes([0.44, 0.035, 0.12, 0.055])
    forward_ax = fig.add_axes([0.61, 0.035, 0.12, 0.055])
    back_button = Button(back_ax, "◀ Step")
    play_button = Button(play_ax, "▶ Play")
    forward_button = Button(forward_ax, "Step ▶")

    state = {"frame": 0, "playing": False}

    def render(frame):
        frame = int(np.clip(frame, 0, n_frames - 1))
        state["frame"] = frame
        for i, (poses, trail, marker, heading_line, ghost) in enumerate(zip(
            robot_poses, trail_lines, robot_markers, heading_lines, ghost_markers
        )):
            pose = poses[frame]
            trail.set_data(poses[:frame + 1, 0], poses[:frame + 1, 1])
            marker.set_data([pose[0]], [pose[1]])
            heading_length = 0.035 * max(float(np.max(span)), 1.0)
            heading_line.set_data(
                [pose[0], pose[0] + heading_length * np.cos(pose[2])],
                [pose[1], pose[1] + heading_length * np.sin(pose[2])],
            )
            ghost_pose = ghost_poses[i][frame]
            if np.all(np.isfinite(ghost_pose[:2])):
                ghost.set_data([ghost_pose[0]], [ghost_pose[1]])
            else:
                ghost.set_data([], [])
        ball = ball_poses[frame]
        ball_trail.set_data(ball_poses[:frame + 1, 0], ball_poses[:frame + 1, 1])
        ball_marker.set_data([ball[0]], [ball[1]])

        robot_ball_ranges = [np.linalg.norm(poses[frame, :2] - ball[:2]) for poses in robot_poses]
        control_text = ", ".join(f"{values[frame]:+.2f}" for values in controls)
        status.set_text(
            f"t = {times[frame]:.2f} s    frame {frame + 1}/{n_frames}\n"
            f"range to ball: R1 {robot_ball_ranges[0]:.1f} | "
            f"R2 {robot_ball_ranges[1]:.1f} | R3 {robot_ball_ranges[2]:.1f}\n"
            f"normal accel [R1, R2, R3]: {control_text} m/s²\n"
            "diamonds = latest computed intercept poses; controllers currently target the ball"
        )
        fig.canvas.draw_idle()

    def on_slider(value):
        render(value)

    frame_slider.on_changed(on_slider)

    def step(delta):
        frame_slider.set_val(int(np.clip(state["frame"] + delta, 0, n_frames - 1)))

    back_button.on_clicked(lambda _event: step(-1))
    forward_button.on_clicked(lambda _event: step(1))

    def toggle_play(_event=None):
        state["playing"] = not state["playing"]
        if state["playing"]:
            play_button.label.set_text("❚❚ Pause")
            animation.event_source.start()
        else:
            play_button.label.set_text("▶ Play")
            animation.event_source.stop()

    play_button.on_clicked(toggle_play)

    def on_key(event):
        if event.key == "left":
            step(-1)
        elif event.key == "right":
            step(1)
        elif event.key == " " or event.key == "space":
            toggle_play()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def advance(_frame):
        if state["playing"]:
            next_frame = state["frame"] + 1
            if next_frame >= n_frames:
                next_frame = 0
            frame_slider.set_val(next_frame)

    animation = FuncAnimation(fig, advance, interval=100, blit=False, cache_frame_data=False)
    render(0)
    plt.show()
    return fig, animation

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

    terminal_heading = 4.422615269729036 + 0*np.random.uniform(np.pi/2, 3*np.pi/2)
    print(f"terminal_heading: {terminal_heading}")
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

        # if np.linalg.norm(ball0[:2] - target[:2]) < 2*R_ball:
        #     print("Ball is close to the goal.")


        # compute the path manager every second for each robot
        if t_check - P.Ts < t < t_check + P.Ts:

            if np.linalg.norm(x1_robot0[:2] - ball0[:2]) < 2*R_robot:
                print("Robot 1 is close to the ball.")
            if np.linalg.norm(x2_robot0[:2] - ball0[:2]) < 2*R_robot:
                print("Robot 2 is close to the ball.")
            if np.linalg.norm(x3_robot0[:2] - ball0[:2]) < 2*R_robot:
                print("Robot 3 is close to the ball.")

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
            if x1_best_intercept_pose is not None:
                x1_control = pronav(x1_robot.x, x1_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
            if x2_best_intercept_pose is not None:
                x2_control = pronav(x2_robot.x, x2_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
            if x3_best_intercept_pose is not None:
                x3_control = pronav(x3_robot.x, x3_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
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

        if x1_best_intercept_pose is None:
            x1_control = pronav(x1_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
        if x2_best_intercept_pose is None:
            x2_control = pronav(x2_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
        if x3_best_intercept_pose is None:
            x3_control = pronav(x3_robot.x, ball0, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)

        else:
            x1_control = pronav(x1_robot.x, x1_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
            x2_control = pronav(x2_robot.x, x2_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)
            x3_control = pronav(x3_robot.x, x3_best_intercept_pose, v_robot, v_ball, navigation_constant=3, max_acceleration=a_max_robot)


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
