"""PN ghost assignment when the ball's Dubins terminal heading is unknown.

Only the ball simulator sees ``true_terminal_heading``. Robot planning sees
the ball's measured pose, its starting pose, the goal position, and a bank of
candidate terminal headings. Initial ghosts cover separated hypotheses; later
observations select a candidate path and update the stationary PN ghosts.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import params as P
from dynamics import SimpleDynamics
from intercept_fixed import (find_pronav_intercept,
                             find_pronav_intercept_on_trajectory)
from path_follower import get_dubins_parameters
from path_manager import PathManager, dubins_state_at_time
from pronav import pronav
from sim_assignment_pn import animate_simulation, _pose_for_history


def plot_simulation(times, robot_poses, ball_poses, ghost_poses,
                    update_events, goal_xy, hit_robot, save_path):
    """Plot four snapshots of the robot paths and their assigned PN ghosts."""
    times = np.asarray(times)
    ball_poses = np.asarray(ball_poses)
    robot_poses = [np.asarray(poses) for poses in robot_poses]
    ghost_poses = [np.asarray(poses) for poses in ghost_poses]
    final_time = times[-1]
    first_update = update_events[0][0] if update_events else 0.25 * final_time
    snapshot_times = [0., min(final_time, first_update + 0.15),
                      min(final_time, 0.5 * (first_update + final_time)),
                      final_time]
    indices = [min(np.searchsorted(times, instant), len(times) - 1)
               for instant in snapshot_times]
    colors = ("#d85140", "#147d65", "#4874b8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.83, bottom=0.12, left=0.06, right=0.98,
                        hspace=0.25, wspace=0.08)
    fig.suptitle("Interception with an unknown intruder terminal heading",
                 fontsize=17, fontweight="semibold", y=0.98)
    fig.text(0.5, 0.93, "Four moments from the same run  |  diamonds: stationary PN ghosts",
             ha="center", fontsize=11)

    positions = np.vstack([ball_poses[:, :2],
                           *(poses[:, :2] for poses in robot_poses),
                           *(poses[np.isfinite(poses[:, :2]).all(axis=1), :2]
                             for poses in ghost_poses),
                           np.asarray(goal_xy)[None, :]])
    padding = 40.
    xlimits = (np.min(positions[:, 0]) - padding,
               np.max(positions[:, 0]) + padding)
    ylimits = (np.min(positions[:, 1]) - padding,
               np.max(positions[:, 1]) + padding)

    for panel, (ax, index) in enumerate(zip(axes.flat, indices)):
        current_t = times[index]
        if panel == 0:
            stage = "Initial heading assignments"
        elif panel == 1 and update_events:
            stage = f"After observing intruder: heading {update_events[0][1]:.3f} rad"
        elif panel == 3 and hit_robot is not None:
            stage = f"Defender {hit_robot} intercepts"
        else:
            stage = "Continuing toward assigned ghosts"
        ax.set_title(f"t = {current_t:.2f} s  |  {stage}", fontsize=11)
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               ncol=6, frameon=False, fontsize=10)
    fig.text(0.5, 0.04,
             "Pale paths show the completed run, for display only. Dashed lines "
             "point to current ghosts; crosses and arrows show their updates.",
             ha="center", fontsize=10)
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)


def sim(animate=True, update_ghosts=True,
        candidate_headings=None, true_terminal_heading=4.422615269729036,
        update_interval=1.0, error_threshold=15.0, plot=True,
        plot_path="sim_assignment_unknown_heading.png"):
    """Run the three-robot simulation without revealing the terminal heading.

    ``candidate_headings`` must span the ball paths the robots consider
    possible. ``error_threshold`` is measured position disagreement between
    the selected candidate and the ball, in the simulation's distance units.
    """
    if candidate_headings is None:
        # The example's true heading lies in this interval. Replace the bank
        # with your assumed range; a heading outside it cannot be identified.
        candidate_headings = np.linspace(np.pi / 2, 3 * np.pi / 2, 11)
    candidate_headings = np.asarray(candidate_headings, dtype=float)
    if (candidate_headings.ndim != 1 or len(candidate_headings) == 0 or
            not np.all(np.isfinite(candidate_headings))):
        raise ValueError("candidate_headings must be a nonempty finite 1D array")
    if update_interval <= 0 or error_threshold < 0:
        raise ValueError("update_interval must be positive; error_threshold >= 0")

    ball_initial = np.array([1000., 0., np.pi])
    robot_initial = [np.array([0., 100., 0.]),
                     np.array([0., 0., 0.]),
                     np.array([0., -100., 0.])]
    goal_xy = np.array([0., 0.])
    true_target = np.r_[goal_xy, true_terminal_heading]
    v_robot, v_ball = 30., 100.
    a_max_robot, a_max_ball = 2 * P.gravity, P.max_acceleration
    hit_radius, horizon = 10., 100.

    # Only the simulator constructs the real ball path. The assignment code
    # below uses goal_xy and the sampled headings, never true_target[2].
    ball = SimpleDynamics(ball_initial.copy(), v_ball, a_max_ball)
    ball_manager = PathManager(
        v_ball**2 / a_max_ball, ball_initial[:2], ball_initial[2],
        true_target[:2], true_target[2],
    )
    robots = [SimpleDynamics(x.copy(), v_robot, a_max_robot)
              for x in robot_initial]

    def candidate_goal(index):
        return np.r_[goal_xy, candidate_headings[index]]

    # Initially no observed motion distinguishes these models, so distribute
    # robots across the bank. A robot's PN ghost is a fixed point on its
    # assigned candidate ball trajectory, chosen by PN rollout.
    representative_indices = np.rint(
        np.linspace(0, len(candidate_headings) - 1, len(robots))
    ).astype(int)
    candidates = []
    costs = np.full((len(robots), len(robots)), 1e6)
    for i, robot in enumerate(robots):
        row = []
        for j, heading_index in enumerate(representative_indices):
            result = find_pronav_intercept(
                robot.x, ball_initial, candidate_goal(heading_index),
                v_robot, v_ball, a_max_robot, a_max_ball, horizon,
                hit_radius=hit_radius, ghost_dt=0.1, sim_dt=0.02,
            )
            row.append(result)
            if result is not None:
                costs[i, j] = result[0]
        candidates.append(row)
    rows, columns = linear_sum_assignment(costs)
    initial_indices = np.zeros(len(robots), dtype=int)
    ghosts = [None] * len(robots)
    print("Initial Dubins heading hypotheses and PN ghosts:")
    for i, j in zip(rows, columns):
        heading_index = representative_indices[j]
        result = candidates[i][j]
        ghost = None if result is None else result[1].copy()
        initial_indices[i] = heading_index
        ghosts[i] = ghost
        print(f"  Defender {i + 1}: heading={candidate_headings[heading_index]:.3f}, "
              f"ghost={None if ghost is None else ghost[:2].round(2)}, "
              f"predicted hit={None if result is None else round(result[0], 2)}")

    # Compute once: each hypothesis is a complete trajectory from the
    # *original* ball start. Never ask for a fresh shortest Dubins path from
    # the observed position, which can switch to a different path family.
    ball_radius = v_ball**2 / a_max_ball
    candidate_arrivals = []
    for j in range(len(candidate_headings)):
        path = get_dubins_parameters(
            ball_initial[:2], ball_initial[2], goal_xy,
            candidate_headings[j], ball_radius,
        )
        first_arc, second_arc, straight = path[-3:]
        candidate_arrivals.append(
            (ball_radius * (first_arc + second_arc) + straight) / v_ball
        )

    active_heading_index = None
    update_events = []
    times = [0.0]
    robot_history = [[robot.x.copy()] for robot in robots]
    ball_history = [ball.x.copy()]
    ghost_history = [[_pose_for_history(ghost)] for ghost in ghosts]
    control_history = [[np.nan] for _ in robots]
    hit_robot = None
    hit_target = False
    t = 0.0
    update_steps = max(1, int(round(update_interval / P.Ts)))

    for step in range(int(horizon / P.Ts)):
        t = step * P.Ts
        if update_ghosts and step and step % update_steps == 0:
            # Rank the complete candidate trajectories by their agreement
            # with the ball's observed position at the same absolute time.
            candidate_states = [
                dubins_state_at_time(
                    ball_initial, candidate_goal(j), t, v_ball, a_max_ball,
                )
                for j in range(len(candidate_headings))
            ]
            errors = np.array([
                np.linalg.norm(ball.x[:2] - state[:2])
                for state in candidate_states
            ])
            best_error = float(np.min(errors))
            tied_indices = np.flatnonzero(
                np.abs(errors - best_error) <= 1e-6
            )
            if len(tied_indices) == 1:
                chosen = int(tied_indices[0])
            elif active_heading_index in tied_indices:
                # Keep the current hypothesis while it remains tied for best.
                chosen = active_heading_index
            else:
                # Resolve a positional tie toward the center of the heading
                # bank. For a bank symmetric about pi, this picks mirrored
                # hypotheses (e.g. 1.885 and 4.398 rad) instead of whichever
                # candidate happens to appear first in the array.
                bank_center = 0.5 * (
                    np.min(candidate_headings) + np.max(candidate_headings)
                )
                center_distances = np.abs(
                    candidate_headings[tied_indices] - bank_center
                )
                nearest_to_center = tied_indices[
                    np.flatnonzero(
                        np.abs(center_distances - np.min(center_distances))
                        <= 1e-12
                    )
                ]
                chosen = (int(nearest_to_center[0])
                          if len(nearest_to_center) == 1 else None)
            active_error = (np.inf if active_heading_index is None
                            else errors[active_heading_index])

            # Replan when evidence changes the likely heading or the active
            # forecast becomes inaccurate. Keep old ghosts only if a tie also
            # remains symmetric around the bank center.
            should_update = (chosen is not None and
                             (chosen != active_heading_index or
                              active_error > error_threshold))
            if chosen is None:
                print(f"t={t:.2f} s: ambiguous between sampled headings "
                      f"{candidate_headings[tied_indices].round(3)} rad; "
                      "keeping current ghost assignments")
            elif should_update and t < candidate_arrivals[chosen]:
                end = min(candidate_arrivals[chosen], horizon)
                future_times = t + np.arange(
                    int(np.floor((end - t) / P.Ts)) + 1
                ) * P.Ts
                forecast = np.array([
                    dubins_state_at_time(
                        ball_initial, candidate_goal(chosen), future_t,
                        v_ball, a_max_ball,
                    )
                    for future_t in future_times
                ])
                # Anchor the candidate's remaining segment to the current
                # measurement. This corrects its present position error but
                # retains its path shape and time-indexed heading hypothesis.
                forecast[:, :2] += ball.x[:2] - forecast[0, :2]
                for i, robot in enumerate(robots):
                    result = find_pronav_intercept_on_trajectory(
                        robot.x, forecast, v_robot, a_max_robot,
                        hit_radius=hit_radius, ghost_dt=0.1,
                    )
                    if result is not None:
                        ghosts[i] = result[1].copy()
                active_heading_index = chosen
                update_events.append((t, candidate_headings[chosen], errors[chosen]))
                print(f"t={t:.2f} s: closest sampled heading "
                      f"{candidate_headings[chosen]:.3f} rad, "
                      f"intruder-position residual {errors[chosen]:.2f}")
                if errors[chosen] > error_threshold:
                    print("  Intruder motion does not fit any sampled heading well; "
                          "consider widening or refining the bank.")
                for i, ghost in enumerate(ghosts, 1):
                    print(f"  Defender {i} updated ghost: "
                          f"{None if ghost is None else ghost[:2].round(2)}")

        robot_controls = [
            pronav(robot.x, ghost, v_robot, ghost_speed=0.0,
                   max_acceleration=a_max_robot)
            if ghost is not None else np.array([0.0])
            for robot, ghost in zip(robots, ghosts)
        ]
        ball.update(ball_manager.update(ball))
        for robot, control in zip(robots, robot_controls):
            robot.update(control)
        t = (step + 1) * P.Ts

        times.append(t)
        ball_history.append(ball.x.copy())
        for i, (robot, ghost, control) in enumerate(
            zip(robots, ghosts, robot_controls)
        ):
            robot_history[i].append(robot.x.copy())
            ghost_history[i].append(_pose_for_history(ghost))
            control_history[i].append(float(control[0]))

        for i, robot in enumerate(robots):
            if np.linalg.norm(robot.x[:2] - ball.x[:2]) < hit_radius:
                hit_robot = i + 1
                print(f"Defender {hit_robot} hit the intruder at t={t:.2f} s.")
                break
        if hit_robot is not None:
            break
        if np.linalg.norm(ball.x[:2] - goal_xy) < 5.0:
            hit_target = True
            print(f"Intruder reached the goal at t={t:.2f} s.")
            break

    if plot:
        plot_simulation(times,
                        [np.asarray(history) for history in robot_history],
                        ball_history,
                        [np.asarray(history) for history in ghost_history],
                        update_events, goal_xy, hit_robot, plot_path)
        print(f"Saved static plot to {plot_path}")

    if animate:
        animate_simulation(
            np.asarray(times),
            [np.asarray(history) for history in robot_history],
            np.asarray(ball_history),
            [np.asarray(history) for history in ghost_history],
            [np.asarray(history) for history in control_history],
            true_target,
        )
    return {
        "hit_robot": hit_robot,
        "hit_time": t if hit_robot is not None else None,
        "ball_reached_goal": hit_target,
        "heading_updates": update_events,
        "initial_heading_indices": initial_indices,
    }


if __name__ == "__main__":
    true_terminal_heading = -np.pi/2 + 0*np.random.uniform(np.pi / 2, 3 * np.pi / 2)
    # true_terminal_heading = 4.422615269729036
    print(f"Simulating with true terminal heading {true_terminal_heading:.3f} rad.")
    # sim(true_terminal_heading=true_terminal_heading)
    sim(true_terminal_heading=true_terminal_heading, update_interval=0.5, error_threshold=5.)
    
