"""Run regression checks and legacy Dubins heading scenarios.

Usage:
    python3 run_sim_assignment_checks.py
    python3 run_sim_assignment_checks.py --headings 90 135 270

Headings supplied on the command line are in degrees. Exit status is zero only
when all regression checks pass and every selected heading yields an intercept.
"""

import argparse
import sys
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from sim_assignment import sim


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--headings", nargs="+", type=float, metavar="DEG",
        help="Test these headings instead of the ten documented scenarios.",
    )
    args = parser.parse_args(argv)

    if args.headings is None:
        headings = list(np.linspace(np.pi / 2, 3 * np.pi / 2, 9))
        headings.insert(-1, 4.422615269729036)
    else:
        if not np.all(np.isfinite(args.headings)):
            parser.error("headings must be finite numbers")
        headings = np.deg2rad(args.headings)

    print("Running regression checks...", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_sim_assignment", "test_ball_path_manager",
        "test_assignment_plotting", "test_short_intruder_paths",
    ])
    regression = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)

    print("\nRunning heading scenarios with ball_policy='dubins'.")
    print("PASS means a defender intercepts; MISS means no intercept.")
    interceptions = misses = errors = 0
    for heading in headings:
        degrees = float(np.rad2deg(heading))
        print(f"\nRunning {degrees:.4f} deg...", flush=True)
        try:
            run = sim(terminal_heading=float(heading),
                      plot=False, animate=False, verbose=False, selection_mode="median",
                      ball_policy="dubins")
            closest = min(run["closest_approach"])
            if run["hit_robot"] is not None:
                status = "PASS"
                interceptions += 1
            else:
                status = "MISS"
                misses += 1
            print(f"{degrees:8.4f} deg | {status} | {run['outcome']} | "
                  f"closest = {closest:.2f} m", flush=True)
            timing = run["timing"]
            planning_ms = timing["planning_seconds"] * 1000
            print(f"  Compute: {timing['simulation_seconds']:.3f} s wall | "
                  f"{timing['realtime_factor']:.2f}x real time | "
                  f"planner mean/max: {planning_ms.mean():.2f}/{planning_ms.max():.2f} ms "
                  f"({len(planning_ms)} calls)", flush=True)
        except Exception as exc:
            errors += 1
            print(f"{degrees:8.4f} deg | ERROR | {type(exc).__name__}: {exc}",
                  flush=True)

    regression_passed = (regression.testsRun - len(regression.failures)
                         - len(regression.errors) - len(regression.skipped))
    print(f"\nRegression checks: {regression_passed}/{regression.testsRun} passed.")
    print(f"Heading scenarios: {interceptions}/{len(headings)} intercepted, "
          f"{misses} missed, {errors} errors.")
    return 0 if regression.wasSuccessful() and not misses and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
