"""tello_aruco_landing.Lander end to end on the synthetic Tello: rendered
video, the real estimator/tracker/filter, the real mission, the log."""

import threading

import numpy as np
import pytest

import sim
import tello_map
from landing_control import BOARD
from tello_aruco_landing import LOG_COLS, Lander, parse_turns
from tello_pose import NOMINAL_DIST, NOMINAL_K, BoardGeometry

ROUTE = (4, 5, 6, 7)


def make(stop=None, **kw):
    geom = BoardGeometry()
    sheets, pad = sim.route_scene(geom)
    drone = sim.FakeTello(sim.Scene(sheets.values()))
    lander = Lander(drone, NOMINAL_K, NOMINAL_DIST, waypoints=ROUTE,
                    search_turn=parse_turns("R R R R L", ROUTE), board=geom,
                    stop=stop, now=drone.now, sleep=drone.sleep, **kw)
    return lander, drone, pad


def test_flies_the_route_and_lands_on_the_pad(tmp_path, capsys):
    log = tmp_path / "run.csv"
    lander, drone, pad = make(log=str(log))
    reason = lander.run()
    out = capsys.readouterr().out
    assert reason == "landed on the pad", out[-2000:]
    for w in ROUTE:
        assert f"at marker {w} -> HOVER" in out
    assert out.index("at marker 4") < out.index("at marker 5") < out.index(
        "at marker 6") < out.index("at marker 7") < out.index("HOP forward")
    err = np.hypot(*(drone.landed_at - pad)[[0, 2]])
    assert err < 5.0, f"landed {err:.1f} cm from the pad centre"
    assert [c for _, c in drone.log if c in ("takeoff", "land")] == ["takeoff", "land"]

    # the log replays in tello_map.py, with the full state packet in it
    rows = tello_map.read_log(str(log))
    assert len(rows) > 100
    assert sum(p is not None for _, p, _, _ in rows) > 0.6 * len(rows)
    header = log.read_text().splitlines()[0].split(",")
    assert header == LOG_COLS


def test_stop_lands_at_once():
    stop = threading.Event()
    stop.set()
    lander, drone, _ = make(stop=stop)
    assert lander.run() == "stopped by hand"
    assert [c for _, c in drone.log] == ["takeoff", "land"]


def test_parse_turns():
    assert parse_turns("", ROUTE) == {4: 1, 5: 1, 6: 1, 7: 1, BOARD: -1}
    assert parse_turns("r,L r right LEFT", ROUTE) == {4: 1, 5: -1, 6: 1, 7: 1, BOARD: -1}
    with pytest.raises(ValueError):
        parse_turns("R R", ROUTE)
    with pytest.raises(ValueError):
        parse_turns("R R R R X", ROUTE)
