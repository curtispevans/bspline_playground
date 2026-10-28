# Algorithm in `sim_assignment.py`

This simulation uses three defenders, periodic selection of stationary ghost
targets, and proportional navigation (PN). Each defender chooses its own ghost
independently. By default the moving ball applies maximum left normal
acceleration for two seconds, then switches to PN toward the goal position.
Close to the ball, a defender switches from its ghost to PN against the ball's
measured pose and velocity. Defender forecasts remain hypothetical Dubins
approaches, which can differ from the ball's actual maneuver and PN trajectory.

The planner and guidance remain the simple approach from `sim_assignment.py`.
The plots and returned histories make successes and misses easier to inspect.
This approach does **not yet guarantee an intercept for every terminal heading**.

## Setup and motion model

| Quantity | Default |
| --- | --- |
| Ball start `[x, y, heading]` | `[1000, 0, pi]` |
| Defender starts | `[0, 100, 0]`, `[0, 0, 0]`, `[0, -100, 0]` |
| Ball goal position | `[0, 0]` |
| Ball policy | `"max_accel_pronav"` |
| Ball maneuver duration / turn direction / PN gain | 2 s / +1 (left) / 3 |
| Terminal heading for legacy `ball_policy="dubins"` | `4.422615269729036` rad |
| Defender / ball speed | 30 / 100 m/s |
| Defender / ball maximum normal acceleration | 19.62 / 39.24 m/s² |
| Integration step, `P.Ts` | 0.01 s |
| Ghost update interval | 0.5 s |
| Intercept / goal radius | 10 / 10 m |
| Simulation time limit | 100 s |

`SimpleDynamics` integrates the constant-speed planar model with RK4:

```text
x_dot       = speed * cos(heading)
y_dot       = speed * sin(heading)
heading_dot = normal_acceleration / speed
```

Normal acceleration is limited to the vehicle's configured maximum. Its minimum
turn radius is `R = speed**2 / max_acceleration`: approximately 45.87 m for a
defender and 254.84 m for the ball. Defenders use PN; they do not follow robot
`PathManager` instances.

## Ball guidance

`MaxAccelerationPronavPathManager` in `path_manager.py` has three states:

1. `maneuver`: command `ball_turn_direction * ball.max_acceleration` at every
   integration step. Speed remains constant; +1 turns left (CCW), -1 right.
2. `pronav`: once `ball_maneuver_duration` seconds have elapsed, permanently
   switch to `pronav` against the stationary goal, using
   `ball_navigation_constant` and the ball's acceleration limit. The switch
   occurs on the first control step at or after the duration. Zero duration
   starts directly in PN. The final heading is determined by this flight.
3. `complete`: if the measured position is within `goal_radius` during either
   phase, mark the manager done and return zero normal acceleration. A caller
   using the manager directly should stop integrating when it is done.

For example:

```python
sim(ball_maneuver_duration=2.0, ball_turn_direction=1,
    ball_navigation_constant=3.0)
sim(ball_maneuver_duration=3.0, ball_turn_direction=-1)
sim(ball_maneuver_duration=0.0)  # PN from the start.
sim(ball_policy="dubins", terminal_heading=4.422615269729036)
```

Only the legacy Dubins policy uses `terminal_heading` to control actual ball
motion. Defender planning still samples its own terminal-heading bank using
the measured ball pose and goal position. The actual maneuver settings are
not supplied to that planner. The heading-based unknown-heading demonstration
in `sim_assignment_unknown_heading.py` continues to use its Dubins ball model.

## 1. Pick an initial ghost for each defender

Call `intercept.get_cost_metrics` once with all defender poses. For every
eligible terminal heading, the defender with the shortest predicted intercept
time owns that heading. This divides the heading sweep into regions; headings
are sampled, so each boundary falls between adjacent samples where ownership
changes. When the defender poses and candidate sweep are mirror-symmetric around
the ball-to-goal axis, the centered defender owns the exact midpoint heading.
Choose the target selection rule with `selection_mode`:

| Mode | Target selection |
| --- | --- |
| `"slowest_symmetric"` (default) | Slowest feasible intercept in each partition, with the midpoint heading selected by the centered defender when geometry is symmetric |
| `"median"` | Median feasible intercept time in every partition, including the centered defender; even counts use the lower middle sample |

Both modes use the same short-path filter, cost matrix, and heading ownership.
The mode selects a target after those computations. For example:

```python
sim(selection_mode="slowest_symmetric")
sim(selection_mode="median")
```

The cost calculation currently works as follows:

1. Start with the original **10 hypothetical ball terminal headings within 90
   degrees of the ball's current heading**, including the endpoints, and the
   exact midpoint heading. Add geometric boundary headings for short paths
   consisting of a nonnegative straight segment followed by a terminal turn.
   These are found from radius-R circles tangent to the current heading line
   and passing through the goal, without using the true terminal heading.
   A terminal turn can have a zero-length straight segment. Initially the
   boundary heading duplicates the midpoint, preserving the 11 original
   candidates from 90 to 270 degrees. Later it can add a short approach outside
   the rotating bank, including a narrow terminal-turn branch missed by the
   original 20-degree sampling. Also consider each current minimum-radius turn
   circle: if it passes within the **10 m goal region**, add a terminal arc
   ending at the point on that circle nearest the goal. This preserves the
   configured turn radius when tracking error makes the exact goal point
   slightly inaccessible on the current turn. `sim(goal_radius=...)` uses the
   same radius for candidate generation and goal arrival; standalone
   `get_cost_metrics(goal_radius=0)` requests an exact point goal.
2. For each hypothetical heading, construct a candidate ball goal with the same
   goal position, or the recorded in-region endpoint for a terminal arc.
   This **replaces the supplied goal heading** for that prediction.
   Compute its intruder path length `L_B = R_B * (arc_start + arc_end) + straight`.
   Sort these lengths and retain the lowest cluster before the first gap
   larger than `pi * R_B` (half a turning circumference). Also reject paths
   whose **total turning exceeds pi radians**. This second check prevents
   selecting a long looping route when every sampled path is long and no gap
   separates the candidates. Only retained headings are evaluated for defender
   interception. The eligible heading set is recomputed at every replan;
   supplied custom heading banks receive the same filter without augmentation.
3. `find_intercept` predicts the ball pose at a future time `T` using
   `dubins_state_at_time`. For each candidate, cap the interception search at
   `min(remaining_horizon, intruder_path_length / v_ball)`. The forecast ends
   at the goal; a robot arriving afterward must not receive an intercept by
   treating the clamped goal position as a stationary intruder.
4. At that predicted position, set the defender's desired arrival heading to the
   predicted ball heading **minus pi**, so the planned encounter is head-on.
5. Compute the defender's Dubins path length `L(T)` to that arrival pose using
   the defender's minimum turn radius.
6. Scan future times at nominal 0.1 s spacing for a negative-to-nonnegative
   crossing of `g(T) = v_robot * T - L(T)`. Refine a crossing with Brent's method.
   This equates the defender's available travel distance with its planned path
   length. A search with no detected crossing returns `None`.
7. For each heading, assign it to the defender with the smallest feasible
   predicted intercept time. If no defender can intercept a heading, it remains
   unassigned.
   For a mirror-symmetric team and heading sweep, the defender centered on the
   symmetry axis owns the exact midpoint heading when it is feasible.
8. Each defender chooses a heading according to `selection_mode` and stores its
   encounter pose as a stationary **ghost**. The default selects the slowest
   assigned intercept, except for the symmetric centered defender's midpoint
   heading. Median mode sorts assigned intercept times and chooses the middle
   sample (the lower middle sample for an even count). A defender with no
   assigned feasible heading has no fixed ghost.

When no short candidate reaches the goal point or allowed goal region, all
defenders have no fixed ghost and use
the existing direct-PN fallback. The code does not promote the shortest long
detour to a valid approach. This can occur on a terminal turn when the measured
pose and exact goal cannot be connected by a short path in the current
turn-straight-turn model, including small tracking errors near the goal.

The ghost's heading affects the Dubins planning cost. Its heading does not affect
PN while its assigned speed is zero.

## 2. Refresh the ghosts periodically

At each update deadline, repeat the same calculation from every defender's
current pose and the ball's current measured pose. Use `horizon - t` as the
remaining search horizon.

The default refreshes are at 0.5, 1.0, 1.5, … seconds. If an interval falls between
integration steps, refresh on the first step at or after its deadline. For
example, a 0.025 s interval refreshes at 0.03, 0.05, 0.08, … seconds with a 0.01 s
integration step. A refresh replaces the previous ghost, including clearing it
when no intercept is found.

## 3. Choose guidance independently for every defender

At **every integration step**, each defender makes its own choice:

```text
if no fixed ghost exists OR distance(defender, ball) < R_ball:
    PN target = the ball's current pose
    target speed = v_ball
else:
    PN target = the assigned fixed ghost
    target speed = 0
```

For a closing target, `pronav` commands
`normal_acceleration = N * closing_speed * line_of_sight_rate`, with `N = 3`.
When the target is not closing, the controller uses its existing pure-pursuit
fallback to turn toward it. Commands are saturated at the defender's acceleration
limit.

The guidance mode is reevaluated each step; switching to direct PN is not
permanent. The original refresh-time override within `2 * R_robot` is covered by
the direct-PN rule because `2 * R_robot < R_ball` for these defaults.

## 4. Advance, record, and stop

1. Calculate controls from the current poses.
2. Advance the ball using its path manager and each defender using its PN command.
3. Record the new time, poses, guidance targets, and applied defender commands.
4. Stop if any defender is strictly within `hit_radius` of the ball.
5. Otherwise stop if the ball is strictly within 10 m of the goal.
6. Otherwise continue until the last complete integration step within `horizon`.

An intercept takes priority when an intercept and goal arrival occur in the same
step. Detection uses the sampled positions at the ends of integration steps.

## Plotting and replay

Run the script normally:

```sh
python3 sim_assignment.py
```

It saves `sim_assignment.png` and displays four snapshots: the initial state,
just after the first ghost refresh, an intermediate state, and the final state.
After the snapshot window closes, it opens the interactive replay.

- Colored paths and markers show the defenders; magenta shows the ball.
- Hollow diamonds show selected predicted intercepts. Dashed connectors link
  defenders to active stationary ghosts. In snapshots, dotted connectors show
  direct PN to the moving ball.
- Pale paths show the completed recorded run as visual context only.
- Thin intruder candidate paths form the sampled envelope and use the color of
  their assigned defender. Bold dashed paths show each defender's selected
  forecast; gray dotted paths have no feasible owner. The envelope uses the same
  sampled terminal headings, goal, turn radius, and remaining horizon as the
  planner. It is a family of candidate Dubins paths, not an exhaustive reachable
  region.
- The envelope includes only retained short paths. Rejected long candidates
  are marked with crosses and hatched bands in the cost panel. Insets show the
  retained candidate count. Rejected headings have infinite defender costs and
  never receive a ghost.
- Every snapshot uses the recorded plan active for that guidance step. Its
  inset lists the planning time, assigned path counts, and selected terminal
  headings and predicted interception times.
- The replay includes defender heading indicators, a frame slider, step buttons,
  and Play/Pause. Left/right arrows step through frames; Space toggles playback.
- The replay also shows the intercept-time cost curves. Shaded heading bands
  indicate ownership, opaque points indicate assigned samples, and diamonds
  identify the selected targets. Titles, legends, and insets identify the mode
  and label midpoint overrides. Both this panel and the spatial envelope update when
  the slider crosses a replan, including when stepping backward. Selected
  predicted intercepts remain visible when guidance switches to direct PN.
- The replay status shows time, defender-to-ball distances, and normal
  accelerations. Playback starts paused. Its 100 ms frame interval makes it
  slower than simulation time at the default 0.01 s integration step.

To display snapshots without saving, pass `plot_path=None`. To show only replay,
pass `plot=False`. Interactive controls require an interactive Matplotlib backend
and a plot window; an inline/static image cannot respond to replay controls.

Run `python3 intercept.py` to inspect the initial candidate envelope beside its
cost curves, using the same defender numbering and formation as the simulation.
Its final `test(selection_mode=...)` call offers the same two choices.

```python
from sim_assignment import sim

result = sim(ball_maneuver_duration=2.0, ball_turn_direction=1,
             update_interval=0.5,
             plot=True, animate=True)
```

For a run without figures:

```python
result = sim(plot=False, animate=False, verbose=False)
print(result["outcome"])
print(result["closest_approach"])
```

The returned dictionary includes:

| Key | Meaning |
| --- | --- |
| `selection_mode` | Target selection rule used for the run |
| `ball_policy` | Actual ball policy: `max_accel_pronav` or `dubins` |
| `ball_controls` | Ball normal acceleration for the preceding integration interval; initial value is NaN |
| `ball_path_states` | Ball manager state for the preceding integration interval; initial value is `initial` |
| `ball_switch_time` | Elapsed time of first PN command; `None` if PN was never used or policy is Dubins |
| `hit_robot`, `hit_time` | Defender number (1–3) and intercept time; `None` on a miss |
| `ball_reached_goal`, `outcome` | Goal-arrival flag and readable termination reason |
| `times`, `ball_poses` | Recorded time and ball pose, including time zero |
| `robot_poses` | One pose array per defender, aligned with `times` |
| `ghost_poses` | Active fixed ghost per recorded step; NaNs mean direct PN |
| `controls` | Acceleration applied during the preceding integration interval; initial value is NaN |
| `ranges` | Sampled ball distance per defender, aligned with `times` |
| `closest_approach`, `closest_approach_times` | Minimum sampled range and its time for each defender |
| `replan_times` | Actual ghost-refresh times, excluding initialization |
| `assignment_history` | Initial plan and replans, with times, candidate headings, exact ownership (`-1` means unassigned), full cost matrices, selected indices and ghosts, and forecast inputs |
| `timing` | Wall-clock compute duration, individual planner durations, initial/replan statistics, real-time factor, and separate display-call durations; all durations are seconds |
| `replay` | Figure/animation pair when replay is requested; otherwise `None` |

Timing is printed with `verbose=True`. For measurements without console or
figure overhead, run with `plot=False, animate=False, verbose=False` and inspect
`result["timing"]`. `simulation_seconds` covers setup, planning, guidance,
dynamics, and recording, ending before any figures are created. It excludes
module imports. `planning_seconds[0]` is the initial assignment; subsequent
entries align with `replan_times`. Each recorded plan also stores its own
`compute_seconds`. `realtime_factor` is simulated seconds divided by compute
seconds, so values greater than one indicate faster-than-real-time computation.
`plot_seconds` and `animation_seconds` measure the display calls, including any
time spent waiting for windows to close; they are zero when disabled.
The check runner prints compute and planner timings for each heading scenario.

Each recorded plan also includes `eligible_mask`, `intruder_path_lengths`,
`intruder_path_turns`, and `short_path_length_cutoff` for inspecting the filter.
An infinite length cutoff means no large gap was found; the turning restriction
still applies. The full candidate list is retained for diagnostics, including
rejected headings, while only eligible headings participate in assignment.
`candidate_goals` records the actual endpoint of each forecast, and
`terminal_turn_mask` identifies terminal-arc candidates ending in the goal region.
Each plan records `selection_mode` and `selection_reasons` (`slowest`, `symmetry`,
`median`, or `unassigned`) so replay uses the rule that produced that plan.

Ghosts recorded after a step are the guidance targets used during that step. A
refresh at time `t` therefore first appears in the history at `t + P.Ts`. Direct
PN follows the current moving ball; its target is not stored as a fixed diamond.

## Reliability fixes included with the plotting work

- Choose each defender's control independently. The previous `else` belonged
  only to Robot 3's condition and could overwrite Robots 1 and 2's direct-PN
  commands, or try to guide toward their missing ghosts.
- Remove unused defender path managers and handle missing ghosts throughout
  recording and display.
- Schedule refreshes by crossing a deadline rather than a floating-point time
  window. Limit each planning call to the remaining simulation time.
- Record outcome, range, controls, and time zero so misses can be reproduced and
  inspected.

## What remains before expecting an intercept every time

1. **The predicted ball path can differ from the actual path.** The real ball
   uses a timed maximum-acceleration maneuver followed by PN; the cost metric
   searches its own Dubins heading bank. A ghost on a hypothetical path may
   never be visited by the real ball. Planning against the new policy would
   require candidate maneuver durations and directions, with PN rollouts after
   each candidate switch. The legacy Dubins mode also uses a specific actual
   terminal heading that can differ from the planner's hypotheses.
2. **The planning controller and executed controller differ.** The cost predicts
   a Dubins traversal with a required terminal heading; execution uses saturated
   PN toward a fixed point and later switches to the moving ball. A Dubins
   intercept time is not a verified PN intercept. Validate the selected ghost
   with the same PN law, saturation, and switching rule used by the simulation.
3. **The planner can find a meeting after goal arrival.** Its analytic ball
   prediction is clamped at the goal after the candidate's arrival, but the
   simulation stops when the real ball reaches the goal. Restrict any reliability
   search to the available time before goal arrival.
4. **Replanning can move a useful ghost.** Inspect ghost changes and range near a
   miss before adjusting the update interval or the direct-PN switching radius.
5. **A universal claim needs a defined scenario set.** Fixed speeds and turn
   limits make some initial conditions infeasible. Test a stated set of starts,
   headings, and limits, and retain the miss cases as regressions.

Run the regression suite, including the new ball-policy checks, and all ten
documented legacy Dubins heading scenarios with:

```sh
python3 run_sim_assignment_checks.py
```

The heading sweep explicitly uses `ball_policy="dubins"`, disables figures,
and prints each outcome and closest range. It tests nine evenly spaced headings
from 90 to 270 degrees plus `4.422615269729036` radians.
The full sweep can take a minute or two. It exits with status 0 only when the
regression checks pass and every selected scenario intercepts; a miss or error
produces status 1.

To select a subset of headings, supply them in degrees. The regression checks
still run:

```sh
python3 run_sim_assignment_checks.py --headings 90 135 270
```

The saved rerun on 2026-10-21 predates dynamic short-path filtering. It used a heading
bank centered on the ball's measured heading and all other defaults unchanged.
Five of these ten cases intercepted; the other five reached the goal. The
previously recorded baseline intercepted three of ten. Closest range means the
smallest sampled distance across all three defenders before termination.

| Actual terminal heading | Outcome | Closest range |
| --- | --- | --- |
| 90° | Goal reached at 11.82 s | 79.80 m |
| 112.5° | Goal reached at 10.72 s | 28.24 m |
| 135° | Defender 3 intercepts at 7.66 s | 9.73 m |
| 157.5° | Defender 2 intercepts at 7.64 s | 8.81 m |
| 180° | Defender 2 intercepts at 7.62 s | 9.44 m |
| 202.5° | Defender 2 intercepts at 7.64 s | 9.00 m |
| 225° | Defender 1 intercepts at 7.66 s | 9.73 m |
| 247.5° | Goal reached at 10.72 s | 28.24 m |
| 253.4° (default, `4.422615269729036` rad) | Goal reached at 10.96 s | 30.17 m |
| 270° | Goal reached at 11.82 s | 79.80 m |

Run the focused regression checks with:

```sh
MPLBACKEND=Agg python3 -m unittest test_ball_path_manager test_sim_assignment test_assignment_plotting test_short_intruder_paths -v
```

All three original focused regression checks passed in that earlier rerun: independent
defender guidance, recording when no ghost is feasible, and replanning at
deadlines between integration steps. All ten heading scenarios completed
without runtime errors.

Before selection modes were added, the short-path update passed 14 focused checks, including the long branch at
2.01 s for 90°, reflected 270° geometry, keeping a terminal heading outside the
old grid, single-arc terminal forecasts inside the goal region, exact-point
fallback with no short candidate, limiting interception to the intruder's
forecast arrival time, and replaying backward across replans.
In those median-mode 90° and 270° runs, both keep 10 of 12 candidates at 2 s, four at
6 s, and one terminal arc from 7 s onward. Both still reach the goal at 11.82 s
with a closest range of 50.31 m; removing long forecasts has not solved the
remaining interception failure.

The current focused suite also checks that both selection modes keep the same
partition and filtering, restore the symmetric midpoint override, use the
slowest intercept for asymmetric geometry, reject unknown modes, and pass the
chosen mode through initialization and every replan.
