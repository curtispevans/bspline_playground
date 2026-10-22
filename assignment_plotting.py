"""Display the candidate intruder paths and the planner's actual assignments."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from path_manager import dubins_state_at_time, get_dubins_parameters


DEFENDER_COLORS = ("#d85140", "#147d65", "#4874b8")
UNASSIGNED_COLOR = "#858585"


def assignment_at_time(history, instant):
    """Use the plan that supplied the guidance recorded at this instant."""
    if not history:
        return None
    effective_times = [plan.get("effective_time", plan.get("time", 0.0))
                       for plan in history]
    index = np.searchsorted(effective_times, instant + 1e-9, side="right") - 1
    return history[index] if index >= 0 else None


def assignment_paths(plan):
    """Sample eligible Dubins paths to the goal or the planning horizon.

    These curves are the sampled intruder envelope used by the planner. Cache
    them in the recorded plan so snapshots and replay share the same curves.
    """
    if "trajectories" not in plan:
        paths = []
        radius = plan["v_ball"]**2 / plan["a_max_ball"]
        eligible = plan.get("eligible_mask", np.ones(len(plan["terminal_thetas"]), dtype=bool))
        for j, theta in enumerate(plan["terminal_thetas"]):
            if not eligible[j]:
                paths.append(np.empty((0, 3)))
                continue
            if "candidate_goals" in plan:
                goal = plan["candidate_goals"][j].copy()
            else:
                goal = plan["goal"].copy()
                goal[2] = theta
            params = get_dubins_parameters(
                plan["ball_pose"][:2], plan["ball_pose"][2],
                goal[:2], theta, radius,
            )
            first_arc, second_arc, straight = params[-3:]
            arrival = (radius * (first_arc + second_arc) + straight) / plan["v_ball"]
            sample_times = np.linspace(0.0, min(arrival, plan["horizon"]), 160)
            paths.append(np.array([
                dubins_state_at_time(plan["ball_pose"], goal, t,
                                     plan["v_ball"], plan["a_max_ball"])
                for t in sample_times
            ]))
        plan["trajectories"] = paths
    return plan["trajectories"]


def assignment_positions(history):
    """Return candidate positions for plot bounds, including later replans."""
    paths = [path[:, :2] for plan in (history or [])
             for path in assignment_paths(plan)]
    return np.vstack(paths) if paths else np.empty((0, 2))


def selection_description(plan):
    """Older recordings used median selection before modes were recorded."""
    mode = (plan or {}).get("selection_mode", "median")
    return "slowest with symmetry" if mode == "slowest_symmetric" else "median"


def selection_reason(plan, robot_index):
    reasons = plan.get("selection_reasons")
    reason = reasons[robot_index] if reasons is not None else "median"
    return "midpoint" if reason == "symmetry" else reason


def assignment_legend(plan=None):
    return [
        Line2D([], [], color="0.45", lw=1, alpha=0.6,
               label="Assigned intruder candidates (defender colors)"),
        Line2D([], [], color="0.25", lw=2.5, ls="--",
               label=f"Selected paths: {selection_description(plan)}"),
        Line2D([], [], color=UNASSIGNED_COLOR, lw=1.2, ls=":",
               label="Unassigned intruder candidates"),
        Line2D([], [], marker="D", ls="None", markerfacecolor="white",
               markeredgecolor="0.25", markersize=7,
               label="Selected predicted intercepts"),
        Line2D([], [], marker="x", ls="None", color="0.4", markersize=7,
               label="Rejected long routes (cost panel)"),
    ]


def draw_assignment_paths(ax, plan):
    """Draw recorded ownership and selected paths; return removable artists."""
    paths = assignment_paths(plan)
    artists = []
    for path, owner in zip(paths, plan["owners"]):
        if not len(path):
            continue
        color = DEFENDER_COLORS[owner % len(DEFENDER_COLORS)] if owner >= 0 else UNASSIGNED_COLOR
        artists.extend(ax.plot(path[:, 0], path[:, 1], color=color,
                               alpha=0.4 if owner >= 0 else 0.65,
                               lw=1.1, ls="-" if owner >= 0 else ":", zorder=1))
    for robot_id, selected_j in enumerate(plan["selected_indices"]):
        if selected_j < 0:
            continue
        color = DEFENDER_COLORS[robot_id % len(DEFENDER_COLORS)]
        path = paths[selected_j]
        artists.extend(ax.plot(path[:, 0], path[:, 1], color=color,
                               lw=2.5, ls="--", alpha=0.95, zorder=1.5))
        pose = plan["selected_poses"][robot_id]
        artists.extend(ax.plot(pose[0], pose[1], marker="D", ls="None",
                               markerfacecolor="white", markeredgecolor=color,
                               markeredgewidth=1.8, markersize=7, zorder=5))
    return artists


def assignment_summary(plan):
    rows = [f"Plan at t = {plan.get('time', 0.0):.2f} s"]
    if "eligible_mask" in plan:
        count = int(np.count_nonzero(plan["eligible_mask"]))
        terminal = int(np.count_nonzero(plan["eligible_mask"] &
                       plan.get("terminal_turn_mask", np.zeros(len(plan["owners"]), dtype=bool))))
        rows.append(f"Short paths: {count}/{len(plan['terminal_thetas'])} | terminal arcs: {terminal}")
    for i, selected_j in enumerate(plan["selected_indices"]):
        count = int(np.count_nonzero(plan["owners"] == i))
        if selected_j < 0:
            rows.append(f"D{i + 1}: no feasible assigned heading")
        else:
            theta = np.rad2deg(plan["terminal_thetas"][selected_j])
            tau = plan["intercept_times"][i, selected_j]
            reason = selection_reason(plan, i)
            rows.append(f"D{i + 1}: {count} paths | {reason} {theta:.1f}° / {tau:.2f} s")
    eligible = plan.get("eligible_mask", np.ones(len(plan["owners"]), dtype=bool))
    unassigned = int(np.count_nonzero((plan["owners"] < 0) & eligible))
    if unassigned:
        rows.append(f"Unassigned: {unassigned} paths")
    return "\n".join(rows)


def draw_assignment_costs(ax, plan):
    """Show each cost curve, owned heading samples, and the selected target."""
    headings = np.rad2deg(plan["terminal_thetas"])
    owners = plan["owners"]
    eligible = plan.get("eligible_mask", np.ones(len(headings), dtype=bool))
    if len(headings):
        order = np.argsort(headings)
        ordered_headings = headings[order]
        boundaries = np.r_[ordered_headings[0],
                           0.5 * (ordered_headings[:-1] + ordered_headings[1:]),
                           ordered_headings[-1]]
        for k, j in enumerate(order):
            owner = owners[j]
            color = DEFENDER_COLORS[owner % len(DEFENDER_COLORS)] if owner >= 0 else UNASSIGNED_COLOR
            ax.axvspan(boundaries[k], boundaries[k + 1], color=color,
                       alpha=0.09 if eligible[j] else 0.18,
                       hatch=None if eligible[j] else "///", lw=0)
        if np.any(~eligible):
            ax.plot(headings[~eligible], np.full(np.count_nonzero(~eligible), 0.96),
                    marker="x", ls="None", color="0.4", markersize=6,
                    transform=ax.get_xaxis_transform(),
                    label="Rejected long intruder routes")

    for i, times in enumerate(plan["intercept_times"]):
        color = DEFENDER_COLORS[i % len(DEFENDER_COLORS)]
        finite = np.isfinite(times)
        ax.plot(headings, np.where(finite, times, np.nan), color=color,
                alpha=0.3, lw=1.2)
        assigned = (owners == i) & finite
        ax.plot(headings, np.where(assigned, times, np.nan), color=color,
                marker="o", markersize=5, lw=2,
                label=f"Defender {i + 1}: {np.count_nonzero(assigned)} paths")
        selected_j = plan["selected_indices"][i]
        if selected_j >= 0:
            ax.plot(headings[selected_j], times[selected_j], marker="D",
                    markersize=9, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=2, zorder=6)
            reason = selection_reason(plan, i)
            on_right = headings[selected_j] > 0.5 * (headings.min() + headings.max())
            ax.annotate(f"D{i + 1} {reason}: {headings[selected_j]:.1f}°\n{times[selected_j]:.2f} s",
                        (headings[selected_j], times[selected_j]),
                        xytext=(-7 if on_right else 7, 12), textcoords="offset points",
                        ha="right" if on_right else "left",
                        fontsize=9, color=color,
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    if not np.isfinite(plan["intercept_times"]).any():
        message = ("No short intruder paths; direct PN" if not eligible.any()
                   else "No feasible intercepts in this horizon")
        ax.text(0.5, 0.5, message,
                transform=ax.transAxes, ha="center", va="center")
        ax.set_ylim(0., 1.)
    else:
        feasible_times = plan["intercept_times"][np.isfinite(plan["intercept_times"])]
        padding = 0.08 * max(float(np.ptp(feasible_times)), 0.5)
        ax.set_ylim(max(0., float(np.min(feasible_times)) - padding),
                    float(np.max(feasible_times)) + padding)
    if len(headings):
        padding = max(0.04 * float(np.ptp(headings)), 1.)
        ax.set_xlim(float(np.min(headings)) - padding,
                    float(np.max(headings)) + padding)
    ax.set_title(f"Partition: {selection_description(plan)} | plan t = {plan.get('time', 0.0):.2f} s",
                 fontsize=11)
    ax.set_xlabel("Intruder terminal θ (deg)")
    ax.set_ylabel("Predicted time to intercept (s)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)


def plot_assignment(plan, save_path=None, *, show=True):
    """Compare the spatial intruder envelope with the actual assignment costs."""
    fig, (ax, cost_ax) = plt.subplots(1, 2, figsize=(16, 7),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(top=0.84, bottom=0.23, left=0.06, right=0.98, wspace=0.24)
    fig.suptitle("Intruder envelope and defender assignments", fontsize=17,
                 fontweight="semibold", y=0.97)
    fig.text(0.5, 0.915,
             f"Thin: assigned candidates | dashed: {selection_description(plan)} | diamonds: predicted intercepts",
             ha="center", fontsize=10)
    draw_assignment_paths(ax, plan)
    for i, pose in enumerate(plan["robot_poses"]):
        color = DEFENDER_COLORS[i % len(DEFENDER_COLORS)]
        ax.plot(pose[0], pose[1], "o", color=color, markersize=8,
                label=f"Defender {i + 1}", zorder=7)
        ax.annotate(f"D{i + 1}", pose[:2], xytext=(7, 5),
                    textcoords="offset points", color=color, fontsize=10)
    ax.plot(*plan["ball_pose"][:2], "o", color="#a02085", markersize=9,
            label="Intruder", zorder=7)
    ax.plot(*plan["goal"][:2], "*", color="#262626", markersize=13,
            label="Goal", zorder=7)
    ax.text(0.98, 0.02, assignment_summary(plan), transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right", zorder=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.85"})
    ax.set_title("Sampled intruder path envelope", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(alpha=0.2)
    draw_assignment_costs(cost_ax, plan)
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles=handles + assignment_legend(plan), loc="lower center",
               bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False, fontsize=9)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    return fig
