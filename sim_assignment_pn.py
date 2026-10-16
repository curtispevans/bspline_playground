import numpy as np
import params as P
from dynamics import SimpleDynamics
from intercept_fixed import get_cost_metric
from path_manager import PathManager
from pronav import pronav
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
    labels = ("Defender 1", "Defender 2", "Defender 3")
    n_frames = len(times)
    if n_frames == 0:
        print("No simulation frames were recorded.")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.78)

    all_xy = [ball_poses[:, :2], *(poses[:, :2] for poses in robot_poses), target[None, :2]]
    all_xy = np.vstack(all_xy)
    span = np.ptp(all_xy, axis=0)
    margin = max(0.08 * float(np.max(span)), 20.0)
    ax.set_xlim(np.min(all_xy[:, 0]) - margin, np.max(all_xy[:, 0]) + margin)
    ax.set_ylim(np.min(all_xy[:, 1]) - margin, np.max(all_xy[:, 1]) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Defender and Intruder Simulation — replay controls below")
    ax.grid(True, alpha=0.25)
    ax.plot(target[0], target[1], marker="*", color="black", markersize=14, label="Intruder goal")

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
                         label=f"{label} assigned PN ghost")
        trail_lines.append(trail)
        robot_markers.append(marker)
        heading_lines.append(heading)
        ghost_markers.append(ghost)

    ax.plot(ball_poses[:, 0], ball_poses[:, 1], color="magenta", alpha=0.18, linewidth=1.5)
    ball_trail, = ax.plot([], [], color="magenta", linewidth=2.0, label="Ball trail")
    ball_marker, = ax.plot([], [], "o", color="magenta", markersize=9, label="Ball")
    status = fig.text(0.5, 0.215, "",
                    ha="center", va="center",
                    family="monospace", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8", "pad": 3},
                )
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=8,
        frameon=False,
        columnspacing=1.2,
    )

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
            f"range [D1, D2, D3]: "
            f"[{robot_ball_ranges[0]:.1f}, {robot_ball_ranges[1]:.1f}, "
            f"{robot_ball_ranges[2]:.1f}] m    "
            f"normal accel [D1, D2, D3]: [{control_text}] m/s²"
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
    for index, (hit_time, ghost_pose, ghost_time) in enumerate(assignments, 1):
        ghosts.append(None if ghost_pose is None else ghost_pose.copy())
        if ghost_pose is None:
            print(f"Robot {index}: no PN ghost found on this sampled path.")
        else:
            print(f"Robot {index}: ghost={ghost_pose[:2].round(2)}, "
                  f"predicted hit={hit_time:.2f} s, "
                  f"ball at ghost={ghost_time:.2f} s")

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
