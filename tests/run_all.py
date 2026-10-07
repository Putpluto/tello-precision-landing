"""Run every test script, fastest first; exit non-zero if any fails.

    python tests/run_all.py            # everything (~1-2 min)
    python tests/run_all.py --quick    # skip the closed-loop simulator missions
"""

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
TESTS = [
    ("test_landing_control.py", "control law, filter, tracker, state machine"),
    ("test_room_map.py", "room-map chaining, room PnP, coarse-leg signs"),
    ("test_pose_roundtrip.py", "printed board PDF -> detector -> pose"),
    ("test_calibration.py", "calibrator vs a known camera"),
    ("test_recording.py", "GUI recorder: real-time video, frame-for-frame CSV"),
    ("test_single_marker.py", "one marker as the target: config, approach, hop"),
    ("test_setpoint.py", "3D setpoint: limits, map clicks, hold (incl. one sim flight)"),
    ("test_sim_landing.py", "closed-loop missions in the simulator"),
]


def main():
    quick = "--quick" in sys.argv
    results = []
    for name, what in TESTS:
        if quick and name in ("test_sim_landing.py", "test_setpoint.py"):
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parent,
                           capture_output=True, text=True)
        ok = r.returncode == 0
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name:26s} {time.time() - t0:5.1f} s  {what}")
        if not ok:
            print("      " + "\n      ".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
