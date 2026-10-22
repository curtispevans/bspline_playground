# Algorithm: Assignment with an Unknown Terminal Heading

`sim_assignment_unknown_heading.py` simulates three defenders trying to intercept a moving intruder. The intruder follows a Dubins path to a known goal, but its terminal heading is hidden from the defenders. The simulator knows the true heading to generate motion; the assignment and replanning logic uses only a bank of candidate headings and the intruder's observed position.

## Inputs and setup

1. Define the intruder's initial pose, the defenders' initial poses, the goal position, speeds, acceleration limits, hit radius, and simulation horizon.
2. Build a candidate bank of possible terminal headings. By default, the bank contains 11 evenly spaced headings from \(\pi/2\) to \(3\pi/2\).
3. Construct the intruder's actual Dubins path for the simulator. Keep its true terminal heading out of the defender assignment and observation-matching logic.
4. For every candidate heading, compute its complete Dubins trajectory from the intruder's original start pose and its arrival time at the goal. Later observations are compared against these same trajectories, rather than paths recomputed from the intruder's current pose.

## Initial ghost assignment

1. Select up to three representative headings spread across the candidate bank, one per defender.
2. For each defender and representative heading, run a proportional-navigation (PN) intercept calculation against the corresponding candidate intruder path.
3. Use predicted intercept time as the assignment cost. A failed intercept receives a large cost.
4. Apply the Hungarian algorithm to assign each defender to a distinct representative heading, minimizing the total predicted intercept time.
5. Evaluate the assigned candidate trajectory at the predicted PN hit time and store that encounter pose as the defender's stationary **ghost**. The ghost is a point the defender steers toward while the heading is still uncertain; using the encounter pose avoids placing a ghost at the intruder's start just because that was a useful PN waypoint.

## Simulation and periodic heading selection

At each simulation step:

1. At the configured update interval, evaluate every candidate Dubins trajectory at the current absolute simulation time.
2. Measure each candidate's position error against the intruder's observed position and find the smallest error.
3. Select the best-fitting heading. If several candidates tie within tolerance, keep the currently active heading if it is among them. Otherwise, resolve the tie toward the center of the heading bank. If the tie is still unresolved, treat the observation as ambiguous and retain the current ghost assignments.
4. Replan only if a heading was selected and either it differs from the active heading or the active candidate's position error exceeds `error_threshold`.

## Ghost refresh and motion

When replanning is warranted and the selected candidate has not yet reached the goal:

1. Generate the candidate path's remaining, time-indexed trajectory from the current time to its arrival.
2. Translate that forecast so its current position matches the intruder's current measured position. This corrects position error while retaining the candidate path's shape and timing.
3. For each defender, calculate a PN intercept on the remaining forecast. Replace its ghost when an intercept is found. Otherwise clear its fixed ghost and use direct PN to the intruder's measured pose and heading at every simulation step.
4. Record the selected heading, observation error, and update time. If the residual remains above the threshold, report that the heading bank may need to be widened or refined.

Each step, defenders use PN guidance toward their assigned stationary ghosts, or toward the moving intruder when no fixed ghost yielded an intercept. The intruder and defenders are advanced, their poses and ghosts are recorded, and the simulation checks for termination:

- A defender intercepts when it comes within `hit_radius` of the intruder.
- The intruder reaches its target when it comes within 5 distance units of the goal.
- The simulation also ends when the configured horizon is reached.

## Outputs

The function returns the intercepting defender and time (if any), whether the intruder reached the goal, the sequence of heading updates, and the initial heading assignments. Optional plotting and animation visualize the recorded paths and ghost changes.
