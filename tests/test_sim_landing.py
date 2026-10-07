"""Closed-loop: the real flight loop (tello_aruco_landing.Flight) flying
the simulated Tello, on simulated time. Slower than the unit tests (a few
seconds per mission) because every frame is rendered and run through the
real detector - which is the point.

    python tests/test_sim_landing.py
"""

import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
import sim_eval                                        # noqa: E402


def test_lands_on_pad():
    for seed in (101, 102, 103):
        r = sim_eval.run_trial(dict(scenario=sim_eval.scenario(seed), argv=[]))
        assert r["on_pad"], r
        assert r["pad_err"] < 15.0, r


def test_search_first():
    r = sim_eval.run_trial(dict(scenario=sim_eval.scenario(104, search=True), argv=[]))
    assert r["states"].startswith("SEARCH") and r["on_pad"], r


def test_hold_never_lands():
    r = sim_eval.run_trial(dict(scenario=sim_eval.scenario(105),
                                argv=["--hold", "--max-time", "40"]))
    assert "HOLD" in r["states"] and "HOP" not in r["states"], r
    assert "timeout" in r["reason"], r        # it held until the clock ran out


def test_upside_down_board_is_refused():
    """The recorded flights' failure: the sheet mounted rotated 180 deg.
    The old code flew away from it (x and y negated). The new one must
    recognise it, say so, and never act on it: nothing but the search's
    yaw is ever commanded. (Where it ends up is the simulator's hover
    drift, not control, so position is not what is checked.)"""
    import csv
    with tempfile.TemporaryDirectory() as d:
        log = str(pathlib.Path(d) / "upside.csv")
        r = sim_eval.run_trial(dict(scenario=sim_eval.scenario(106, upside_down=True),
                                    argv=["--max-time", "45", "--log", log]))
        rows = list(csv.DictReader(open(log)))
    assert r["upside_down_seen"] and "upside down" in r["reason"], r
    assert "APPROACH" not in r["states"] and not r["crashed"], r
    moved = [row for row in rows if (row["a"], row["b"], row["c"]) != ("0", "0", "0")]
    assert not moved, f"translated while the board was upside down: {moved[:3]}"
    assert any(row["d"] != "0" for row in rows)   # it did search


def test_refuses_bad_calibration():
    """A calibration that fails the sanity checks must not fly - here the
    one this repo shipped with (focal length ~20% long)."""
    import tello_aruco_landing as tal
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "bad.npz")
        K = np.array([[1127.7, 0, 538.5], [0, 1164.1, 399.4], [0, 0, 1.0]])
        np.savez(path, K=K, dist=np.array([[0.135, -2.419, 0.018, 0.0003, 8.465]]),
                 rms=0.745, image_size=np.array([960, 720]))
        args = tal.build_parser().parse_args(["--sim", "--sim-use-calib", "--calib", path])
        try:
            tal.Flight(args)
        except SystemExit as e:
            assert "refusing" in str(e)
            return
        raise AssertionError("flew on a calibration that fails the checks")


def main():
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print("\nPASS" if not failed else f"\nFAIL ({failed})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
