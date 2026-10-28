import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider

from assignment_plotting import (DEFENDER_COLORS, assignment_at_time,
                                 assignment_legend, assignment_positions,
                                 assignment_summary, draw_assignment_costs,
                                 draw_assignment_paths, selection_description)


def _draw_candidate_paths(ax, candidate_paths):
    """Keep the complete heading bank visible beneath the recorded motion."""
    if candidate_paths:
        ax.add_collection(LineCollection(
            [path[:, :2] for path in candidate_paths],
            colors="#6d7e94", linewidths=1.2, alpha=0.6, zorder=1,
            label="Possible intruder trajectories",
        ))


def plot_simulation(times, robot_poses, ball_poses, ghost_poses,
                    update_events, goal_xy, hit_robot, save_path=None, *,
                    title="Interception with an unknown intruder terminal heading",
                    final_status=None, show=True, assignment_history=None,
                    candidate_paths=None):
    """Plot four snapshots; update events are (time, heading or None, ...).

    ``None`` headings represent periodic ghost replanning without heading
    selection. Set ``save_path=None`` to skip saving, or ``show=False`` to
    render a saved figure without opening a window.
    Recorded assignments add the candidate intruder envelope and selected paths
    from the plan active at each snapshot.
    ``candidate_paths`` supplies a fixed bank of sampled (x, y[, heading])
    trajectories when the simulation does not record assignment partitions.
    """
    times = np.asarray(times)
    ball_poses = np.asarray(ball_poses)
    robot_poses = [np.asarray(poses) for poses in robot_poses]
    ghost_poses = [np.asarray(poses) for poses in ghost_poses]
    candidate_paths = ([] if candidate_paths is None else
                       [np.asarray(path) for path in candidate_paths])
    final_time = times[-1]
    first_update = update_events[0][0] if update_events else 0.25 * final_time
    snapshot_times = [0., min(final_time, first_update + 0.15),
                      min(final_time, 0.5 * (first_update + final_time)),
                      final_time]
    indices = [min(np.searchsorted(times, instant), len(times) - 1)
               for instant in snapshot_times]
    colors = [DEFENDER_COLORS[i % len(DEFENDER_COLORS)]
              for i in range(len(robot_poses))]
    snapshot_plans = [assignment_at_time(assignment_history, times[index])
                      for index in indices]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11) if assignment_history else (15, 9),
                             sharex=True, sharey=True)
    fig.subplots_adjust(top=0.79 if assignment_history else 0.83,
                        bottom=0.12, left=0.06, right=0.98,
                        hspace=0.25, wspace=0.08)
    fig.suptitle(title,
                 fontsize=17, fontweight="semibold", y=0.98)
    subtitle = ("Candidate intruder paths use their assigned defender's color | diamonds: selected intercepts"
                if assignment_history else
                "Gray paths: possible intruder trajectories | diamonds: stationary PN ghosts"
                if candidate_paths else
                "Four moments from the same run | diamonds: stationary PN ghosts")
    fig.text(0.5, 0.93, subtitle,
             ha="center", fontsize=11)

    positions = np.vstack([ball_poses[:, :2],
                           *(poses[:, :2] for poses in robot_poses),
                           *(poses[np.isfinite(poses[:, :2]).all(axis=1), :2]
                             for poses in ghost_poses),
                           assignment_positions([plan for plan in snapshot_plans
                                                 if plan is not None]),
                           *(path[:, :2] for path in candidate_paths),
                           np.asarray(goal_xy)[None, :]])
    padding = 40.
    xlimits = (np.min(positions[:, 0]) - padding,
               np.max(positions[:, 0]) + padding)
    ylimits = (np.min(positions[:, 1]) - padding,
               np.max(positions[:, 1]) + padding)

    for panel, (ax, index) in enumerate(zip(axes.flat, indices)):
        current_t = times[index]
        if panel == 0:
            stage = "Initial ghost assignments"
        elif panel == 3 and final_status is not None:
            stage = final_status.rsplit(" at t=", 1)[0]
        elif panel == 1 and update_events and update_events[0][1] is not None:
            stage = f"After observing intruder: heading {update_events[0][1]:.3f} rad"
        elif panel == 1 and update_events:
            stage = "After periodic ghost replanning"
        elif panel == 3 and hit_robot is not None:
            stage = f"Defender {hit_robot} intercepts"
        else:
            stage = "Guidance toward ghosts or the intruder"
        ax.set_title(f"t = {current_t:.2f} s  |  {stage}", fontsize=11)
        _draw_candidate_paths(ax, candidate_paths)
        plan = snapshot_plans[panel]
        if plan is not None:
            draw_assignment_paths(ax, plan)
            ax.text(0.02, 0.02, assignment_summary(plan), transform=ax.transAxes,
                    fontsize=8, va="bottom", zorder=10,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.85"})
        ax.plot(ball_poses[:, 0], ball_poses[:, 1], color="#d7b6d0",
                lw=1.8, label="Intruder path (hindsight)" if panel == 0 else None)
        ax.plot(ball_poses[:index + 1, 0], ball_poses[:index + 1, 1],
                color="#a02085", lw=2.7, label="Intruder traveled" if panel == 0 else None)
        ax.scatter(*ball_poses[index, :2], color="#a02085", s=82, zorder=9,
                   edgecolor="white", linewidth=1)
        ax.scatter(*goal_xy, marker="*", color="#262626", s=160, zorder=8,
                   label="Goal" if panel == 0 else None)

        for robot_id, (poses, ghosts, color) in enumerate(
                zip(robot_poses, ghost_poses, colors), start=1):
            ax.plot(poses[:, 0], poses[:, 1], color=color, lw=1.5, alpha=0.22)
            ax.plot(poses[:index + 1, 0], poses[:index + 1, 1],
                    color=color, lw=2.2,
                    label=f"Defender {robot_id}" if panel == 0 else None)
            ax.scatter(*poses[index, :2], color=color, s=65, zorder=7,
                       edgecolor="white", linewidth=1)
            ghost_xy = ghosts[index, :2]
            if np.isfinite(ghost_xy).all():
                ax.plot([poses[index, 0], ghost_xy[0]],
                        [poses[index, 1], ghost_xy[1]], color=color,
                        lw=1.1, ls="--", alpha=0.48)
                ax.scatter(*ghost_xy, marker="D", facecolor="white",
                           edgecolor=color, linewidth=2, s=75, zorder=6)
                ax.annotate(f"G{robot_id}", ghost_xy, xytext=(6, 6),
                            textcoords="offset points", fontsize=9, color=color)
            else:
                ax.plot([poses[index, 0], ball_poses[index, 0]],
                        [poses[index, 1], ball_poses[index, 1]],
                        color=color, lw=1.1, ls=":", alpha=0.48)

            # Ghosts recorded at the update time are still the old ghosts;
            # the following time step contains the newly assigned ones.
            if panel == 1 and update_events:
                update_index = min(np.searchsorted(times, first_update), len(times) - 1)
                previous_xy = ghosts[update_index, :2]
                if (np.isfinite(previous_xy).all() and np.isfinite(ghost_xy).all()
                        and np.linalg.norm(ghost_xy - previous_xy) > 1.0):
                    ax.scatter(*previous_xy, marker="x", s=55, color=color,
                               alpha=0.8, zorder=6)
                    ax.annotate("", ghost_xy, previous_xy,
                                arrowprops={"arrowstyle": "->", "color": color,
                                            "lw": 1.3, "alpha": 0.8})

        ax.set_xlim(*xlimits)
        ax.set_ylim(*ylimits)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if assignment_history:
        handles += assignment_legend(assignment_history[0])
        labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               ncol=4 if assignment_history or candidate_paths else 6,
               frameon=False, fontsize=9)
    footer = (f"Thin colored paths: assigned intruder candidates; bold dashed paths: {selection_description(assignment_history[0])}.\n"
              "Pale recorded motion is hindsight. Dashed connectors: active fixed ghosts; dotted connectors: direct PN."
              if assignment_history else
              "Gray curves show all candidate headings from the original intruder start.\n"
              "Pale recorded motion is hindsight. Dashed connectors: active fixed ghosts; dotted connectors: direct PN."
              if candidate_paths else
              "Pale paths show the completed run, for display only. Dashed lines "
              "point to fixed ghosts; dotted lines show direct PN to the intruder.")
    fig.text(0.5, 0.04,
             footer, ha="center", fontsize=9)
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def animate_simulation(times, robot_poses, ball_poses, ghost_poses, controls, target,
                       *, assignment_history=None, show=True, candidate_paths=None):
    """Show a replayable simulation with a frame slider and step controls.

    Use the slider or the left/right arrow keys to inspect individual frames.
    Space toggles playback. The faint lines show each complete trajectory;
    bright lines show the trail up to the selected frame.
    With recorded assignments, the intruder envelope and a cost panel update
    together when stepping across a replan.
    A fixed ``candidate_paths`` bank remains visible throughout playback and
    slider navigation, including paths beyond the end of the actual run.
    """
    colors = [DEFENDER_COLORS[i % len(DEFENDER_COLORS)]
              for i in range(len(robot_poses))]
    labels = [f"Defender {i + 1}" for i in range(len(robot_poses))]
    defender_labels = ", ".join(f"D{i + 1}" for i in range(len(robot_poses)))
    n_frames = len(times)
    if n_frames == 0:
        print("No simulation frames were recorded.")
        return None, None
    candidate_paths = ([] if candidate_paths is None else
                       [np.asarray(path) for path in candidate_paths])

    if assignment_history:
        fig, (ax, cost_ax) = plt.subplots(1, 2, figsize=(17, 8),
                                         gridspec_kw={"width_ratios": [1.5, 1]})
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        cost_ax = None
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.30, top=0.78, wspace=0.25)

    all_xy = [ball_poses[:, :2], *(poses[:, :2] for poses in robot_poses),
              *(poses[np.isfinite(poses[:, :2]).all(axis=1), :2]
                for poses in ghost_poses), target[None, :2],
              assignment_positions(assignment_history),
              *(path[:, :2] for path in candidate_paths)]
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
    _draw_candidate_paths(ax, candidate_paths)
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
                         label=f"{label} assigned PN ghost" if not assignment_history else None)
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
    if assignment_history:
        handles += assignment_legend(assignment_history[0])
        legend_labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles, legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4 if assignment_history else 3,
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

    state = {"frame": 0, "playing": False, "assignment": None}
    assignment_artists = []

    def render(frame):
        frame = int(np.clip(frame, 0, n_frames - 1))
        state["frame"] = frame
        plan = assignment_at_time(assignment_history, times[frame])
        if plan is not None and plan is not state["assignment"]:
            for artist in assignment_artists:
                artist.remove()
            assignment_artists[:] = draw_assignment_paths(ax, plan)
            cost_ax.clear()
            draw_assignment_costs(cost_ax, plan)
            ax.set_title(f"Intruder envelope and recorded motion | plan t = {plan.get('time', 0.0):.2f} s",
                         fontsize=11)
            state["assignment"] = plan
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
        range_text = ", ".join(f"{distance:.1f}" for distance in robot_ball_ranges)
        control_text = ", ".join(f"{values[frame]:+.2f}" for values in controls)
        status.set_text(
            f"t = {times[frame]:.2f} s    frame {frame + 1}/{n_frames}\n"
            f"range [{defender_labels}]: [{range_text}] m\n"
            f"normal accel [{defender_labels}]: [{control_text}] m/s²"
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
    # Keep widgets and animation alive when show() returns immediately (e.g.
    # interactive IDEs). Matplotlib stores weak references to widget callbacks.
    fig._simulation_widgets = (frame_slider, back_button, play_button, forward_button)
    fig._simulation_animation = animation
    render(0)
    if show:
        plt.show()
    return fig, animation

