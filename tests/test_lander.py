"""tello_aruco_landing.Lander end to end on the synthetic Tello: rendered
video, the real estimator/tracker/filter, the real mission, the log."""

import threading

import numpy as np
import pytest

import sim
import tello_map
from landing_control import BOARD
from tello_aruco_landing import LOG_COLS, NUDGE_RC, NUDGE_S, Lander, parse_turns
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



def test_notices_when_the_drone_lands_by_itself(capsys):
    """A weak battery or overheating lands the Tello on its own: the
    mission must end there, not keep "searching" on the floor."""
    lander, drone, _ = make()
    real_state = drone.get_current_state

    def state():
        st = real_state()
        if drone.now() > 12.0:                    # sat down mid-route
            st.update(h=0, tof=10, templ=88, temph=90)
        return st
    drone.get_current_state = state
    assert lander.run() == "the drone landed by itself"
    assert "LANDED BY ITSELF" in capsys.readouterr().out
    assert [c for _, c in drone.log if c in ("takeoff", "land")] == ["takeoff"]


def test_yaw_nudge_overrides_then_hands_back(capsys):
    """A nudge by hand sets the yaw for NUDGE_S, then the mission carries
    on - and still lands on the pad."""
    lander, drone, pad = make()
    sent, real_rc, real_state = [], drone.send_rc_control, drone.get_current_state

    def rc(*a):
        sent.append((drone.now(), a))
        real_rc(*a)

    def state():
        if not lander.nudging and 10.0 <= drone.now() < 10.1:
            lander.nudge_yaw(-1)                  # once, 10 s in
        return real_state()
    drone.send_rc_control, drone.get_current_state = rc, state
    assert lander.run() == "landed on the pad"
    during = [a[3] for t, a in sent if 10.05 < t < 10.0 + NUDGE_S - 0.05]
    after = [a[3] for t, a in sent if 10.0 + NUDGE_S + 0.1 < t < 12.0]
    assert during and all(y == -NUDGE_RC for y in during), during
    assert after and any(y != -NUDGE_RC for y in after)
    assert "yaw nudge left (by hand)" in capsys.readouterr().out

def test_parse_turns():
    assert parse_turns("", ROUTE) == {4: 1, 5: 1, 6: 1, 7: 1, BOARD: -1}
    assert parse_turns("r,L r right LEFT", ROUTE) == {4: 1, 5: -1, 6: 1, 7: 1, BOARD: -1}
    with pytest.raises(ValueError):
        parse_turns("R R", ROUTE)
    with pytest.raises(ValueError):
        parse_turns("R R R R X", ROUTE)
