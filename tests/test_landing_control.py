"""The control law's signs and the mission state machine, on synthetic
poses - no images, no drone."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import sim
from landing_control import (BOARD, Mission, MissionConfig, Obs, ServoLaw, State,
                             clamp_setpoint, for_board)
from tello_pose import BoardGeometry

# camera axes in L for a drone square on to the target, facing it
FACING = np.diag([1.0, -1.0, -1.0])
ROUTE = (4, 5, 6, 7)
TURNS = {4: 1, 5: 1, 6: 1, 7: 1, BOARD: -1}


def pose(R=FACING, resolved=True):
    return SimpleNamespace(R_level_cam=R, resolved=resolved)


# ----------------------------------------------------------------------
# The law (landing_control module docstring)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("target, channel, sign", [
    ((0, 0, 100), 1, +1),       # target nearer the board = ahead -> forward
    ((0, 0, 200), 1, -1),
    ((20, 0, 150), 0, +1),      # +X is the drone's right when it faces the board
    ((-20, 0, 150), 0, -1),
    ((0, 20, 150), 2, +1),      # above -> climb
    ((0, -20, 150), 2, -1),
])
def test_servo_signs(target, channel, sign):
    out = ServoLaw().step(np.array([0.0, 0.0, 150.0]), np.zeros(3), FACING,
                          np.array(target, float), aim=np.zeros(3))
    assert np.sign(out.rc[channel]) == sign
    assert all(out.rc[k] == 0 for k in range(4) if k != channel)


def test_yaw_turns_toward_the_aim():
    p = np.array([0.0, 0.0, 150.0])
    right = ServoLaw().step(p, None, FACING, p, aim=np.array([30.0, 0, 0]))
    left = ServoLaw().step(p, None, FACING, p, aim=np.array([-30.0, 0, 0]))
    assert right.rc[3] > 0 and left.rc[3] < 0


def test_errors_are_in_the_body_frame():
    # facing -X (yawed 90 deg right of square on): a target further along -X
    # is straight ahead, not to the side
    R = sim.camera_axes(90.0)
    out = ServoLaw().step(np.array([0.0, 0, 150]), None, R, np.array([-50.0, 0, 150]),
                          aim=np.array([-200.0, 0, 150]))
    assert out.e_fwd == pytest.approx(50.0)
    assert out.e_lat == pytest.approx(0.0, abs=1e-9)
    assert out.rc[1] > 0 and out.rc[0] == 0


def test_unresolved_pose_holds_the_lateral_axis():
    out = ServoLaw().step(np.array([20.0, 0, 150]), np.zeros(3), FACING,
                          np.array([0.0, 0, 150]), aim=np.zeros(3), lateral=False)
    assert out.rc[0] == 0
    assert math.isnan(out.e_lat) and out.e_lat_abs == pytest.approx(20.0)


def test_clamp_setpoint_keeps_the_board_in_view():
    sp, notes = clamp_setpoint((0, 0, 10))
    assert sp[2] == pytest.approx(50.0) and notes


def test_for_board_scales_the_approach_to_the_target():
    assert for_board(BoardGeometry())[0].standoff_far == 150.0
    small = BoardGeometry(ids=(0,), marker_m=0.07)
    assert for_board(small)[0].standoff_far == 100.0


def test_hop_vector_is_from_the_airframe_centre_to_the_pad():
    m = Mission()                                   # pad 25 cm out, lens 4 cm ahead
    assert m.hop_vector((0, 0, 70), FACING) == pytest.approx((49.0, 0.0))
    fwd, right = m.hop_vector((10, 0, 70), FACING)  # 10 cm right of the axis
    assert fwd == pytest.approx(49.0) and right == pytest.approx(-10.0)


# ----------------------------------------------------------------------
# The mission
# ----------------------------------------------------------------------

def fly(mission, sheets, body, t_max=400.0, dt=1.0 / 15.0, resolved=True, battery=80):
    """Step the mission against perfect localization until it lands.
    Returns the notes it printed and where the drone came down (world cm)."""
    notes, t = [], 0.0
    while t < t_max:
        obs = sim.true_obs(body, sheets[mission.target], t, battery, resolved)
        cmd = mission.step(obs)
        if cmd.note:
            notes.append(cmd.note)
        if cmd.action == "hop":
            body.go(*cmd.hop)
        if cmd.action in ("hop", "land"):
            return notes, body.centre()
        body.rc = cmd.rc
        body.advance(dt)
        t += dt
    raise AssertionError(f"still flying after {t_max} s: {notes[-5:]}")


def test_route_visits_each_waypoint_in_order_then_lands_on_the_pad():
    geom = BoardGeometry()
    sheets, pad = sim.route_scene(geom)
    m = Mission(for_board(geom)[0], waypoints=ROUTE, search_turn=TURNS)
    notes, landed = fly(m, sheets, sim.Body((0, 80, 0), 0.0, flying=True))
    visits = [n for n in notes if "-> HOVER" in n]
    assert visits == [f"at marker {w} -> HOVER 2 s" for w in ROUTE]
    assert any("marker 7 done -> turn left, SEARCH for the board" in n for n in notes)
    assert m.reason == "landed on the pad", notes
    assert np.hypot(*(landed - pad)[[0, 2]]) < 3.0


def test_search_turns_the_configured_way():
    m = Mission(waypoints=(4,), search_turn={4: -1})
    cmd = m.step(Obs(t=0.0))
    assert m.state is State.SEARCH and cmd.rc[3] < 0
    m = Mission(waypoints=(4,))                       # unlisted: right
    assert m.step(Obs(t=0.0)).rc[3] > 0


def test_hover_runs_out_then_searches_for_the_next_target():
    m = Mission(waypoints=(4, 5), search_turn={4: 1, 5: -1})
    m.step(Obs(t=0.0))
    m.state, m.t_state = State.HOVER, 0.0
    m.step(Obs(t=1.0))                                # marker out of view: keep hovering
    assert m.state is State.HOVER and m.target == 4
    cmd = m.step(Obs(t=2.1))
    assert m.state is State.SEARCH and m.target == 5
    assert "turn left" in cmd.note
    assert m.step(Obs(t=2.2)).rc[3] < 0



def test_close_enough_to_a_marker_moves_on_without_settling():
    """Within waypoint_close_cm and still drifting at 30 cm/s: that is
    enough at a waypoint - HOVER after waypoint_close_s, no settle."""
    m = Mission(waypoints=(4,))
    drifting = dict(v=np.array([0.0, 0, 30]), pose=pose())
    m.step(Obs(t=0.0, p=np.array([0.0, 0, 150]), **drifting))
    assert m.state is State.GOTO
    m.step(Obs(t=0.1, p=np.array([0.0, 0, 150]), **drifting))   # 50 cm off: not yet
    m.step(Obs(t=0.5, p=np.array([0.0, 0, 150]), **drifting))
    assert m.state is State.GOTO
    m.step(Obs(t=0.6, p=np.array([10.0, 0, 120]), **drifting))  # ~22 cm off
    assert m.state is State.GOTO
    cmd = m.step(Obs(t=0.6 + MissionConfig.waypoint_close_s + 0.05,
                     p=np.array([10.0, 0, 120]), **drifting))
    assert m.state is State.HOVER and "-> HOVER" in cmd.note


def test_skip_by_hand_moves_to_the_next_target_but_not_past_the_board():
    m = Mission(waypoints=(4, 5), search_turn={5: -1})
    m.step(Obs(t=0.0, p=np.array([0.0, 0, 150]), v=np.zeros(3), pose=pose()))
    assert m.state is State.GOTO and m.target == 4
    cmd = m.skip(1.0)
    assert m.state is State.SEARCH and m.target == 5
    assert cmd.note == "marker 4 skipped by hand -> turn left, SEARCH for marker 5"
    m.skip(2.0)
    assert m.target == BOARD
    cmd = m.skip(3.0)                                 # the board stays
    assert m.target == BOARD and m.state is State.SEARCH and "nothing to skip" in cmd.note

def test_losing_the_target_goes_back_to_search():
    m = Mission(waypoints=(4,))
    m.step(Obs(t=0.0, p=np.array([0.0, 0, 150]), v=np.zeros(3), pose=pose()))
    assert m.state is State.GOTO
    m.step(Obs(t=0.5))
    assert m.state is State.GOTO                      # a dropout: hover through it
    m.step(Obs(t=1.5))
    assert m.state is State.SEARCH



def test_losing_the_board_sweeps_left_then_right_not_a_full_turn():
    m = Mission()
    seen = dict(p=np.array([0.0, 0, 150]), v=np.zeros(3), pose=pose())
    m.step(Obs(t=0.0, **seen))
    assert m.state is State.APPROACH
    m.step(Obs(t=1.5))
    assert m.state is State.LOST                      # not SEARCH
    yaw = {round(t, 1): m.step(Obs(t=1.5 + t)).rc[3] for t in (0.5, 1.9, 2.1, 5.9, 6.1, 9.9)}
    assert yaw[0.5] < 0 and yaw[1.9] < 0               # left 2 s
    assert yaw[2.1] > 0 and yaw[5.9] > 0               # right 4 s: back, then 2 s past
    assert yaw[6.1] < 0 and yaw[9.9] < 0               # left again
    cmd = m.step(Obs(t=12.0, **seen))
    assert m.state is State.APPROACH and "found again" in cmd.note


def test_board_not_back_after_the_sweeps_lands():
    m = Mission()
    m.step(Obs(t=0.0, p=np.array([0.0, 0, 150]), v=np.zeros(3), pose=pose()))
    m.state, m.t_state = State.CLOSE, 0.0
    m.step(Obs(t=1.5))
    assert m.state is State.LOST and m.resume is State.CLOSE
    cmd = m.step(Obs(t=1.5 + MissionConfig.lost_max_s + 0.1))
    assert cmd.action == "land" and m.reason == "board lost"

@pytest.mark.parametrize("obs, why", [
    (Obs(t=0.0, battery=10), "battery 10%"),
    (Obs(t=301.0), "mission timeout 300 s"),
])
def test_lands_on_battery_and_timeout(obs, why):
    m = Mission(waypoints=ROUTE)                      # 120 s + 45 s per waypoint
    m.step(Obs(t=0.0))
    cmd = m.step(obs)
    assert cmd.action == "land" and m.reason == why and m.state is State.DONE
    assert m.step(Obs(t=obs.t + 0.1)).action == "done"


def test_lands_when_a_marker_is_not_found():
    m = Mission(waypoints=(6,))
    m.step(Obs(t=0.0))
    cmd = m.step(Obs(t=MissionConfig.search_max_s + 0.1))
    assert cmd.action == "land" and m.reason == "marker 6 not found"


def hop_from(p, resolved=True):
    m = Mission()
    m.step(Obs(t=0.0))
    m.state = State.HOP
    return m, m.step(Obs(t=0.1, p=np.array(p, float), v=np.zeros(3), pose=pose(resolved=resolved)))


def test_hop_goes_forward_and_left_in_go_xyz_terms():
    m, cmd = hop_from((10, 0, 70))
    assert cmd.action == "hop" and cmd.hop == (49, 10)    # 10 cm LEFT: go's +y
    assert m.state is State.DONE and m.reason == "landed on the pad"


def test_hop_goes_straight_when_the_side_is_unknown():
    _, cmd = hop_from((3, 0, 70), resolved=False)
    assert cmd.hop == (49, 0)


def test_implausible_hop_lands_in_place():
    m, cmd = hop_from((0, 0, 200))
    assert cmd.action == "land" and "not plausible" in m.reason


def test_already_over_the_pad_just_lands():
    m, cmd = hop_from((0, 0, 35))
    assert cmd.action == "land" and m.reason == "landed (already over the pad)"
