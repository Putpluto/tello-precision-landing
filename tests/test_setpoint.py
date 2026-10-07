"""AUTO HOLD at an arbitrary 3D setpoint: the limits it is clamped to, the
map clicks that place it (top-down: x and z, side: z and y), the mission
holding it, and a simulated drone actually flying there.

    python tests/test_setpoint.py
"""

import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
from landing_control import (Mission, MissionConfig, SetpointLimits,  # noqa: E402
                             State, clamp_setpoint)
from tello_map import C_SP, LocalizationMap                            # noqa: E402

LIM = SetpointLimits()


# ----------------------------------------------------------------------
# limits
# ----------------------------------------------------------------------

def test_clamp_leaves_good_points_alone():
    for sp in [(0, 0, 70), (30, 5, 120), (-60, -10, 200), (0, 20, 150)]:
        out, notes = clamp_setpoint(sp, LIM)
        assert np.allclose(out, sp) and not notes, (sp, out, notes)


def test_clamp_range_view_height_floor():
    out, notes = clamp_setpoint((0, 0, 10), LIM)            # into the board
    assert out[2] == LIM.z_min and notes
    out, notes = clamp_setpoint((0, 0, 400), LIM)           # out of marker range
    assert out[2] == LIM.z_max and notes
    out, _ = clamp_setpoint((200, 0, 100), LIM)             # far off to the side
    assert abs(math.degrees(math.atan2(out[0], out[2])) - LIM.max_view_deg) < 1e-6
    out, notes = clamp_setpoint((0, 80, 100), LIM)          # above: board leaves the view
    assert abs(math.degrees(math.atan2(out[1], 100.0)) - LIM.max_elev_deg) < 1e-6
    assert "view" in notes[0]
    out, notes = clamp_setpoint((0, -80, 250), LIM, pad_drop=40.0)   # into the floor
    assert out[1] == -40.0 + LIM.floor_clear and "floor" in notes[0]


# ----------------------------------------------------------------------
# map: clicks and drawing share one transform
# ----------------------------------------------------------------------

def test_map_click_roundtrip():
    lm = LocalizationMap()
    size = (470, 600, 4)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, z = rng.uniform(-80, 80), rng.uniform(0, 240)
        hit = lm.to_world(*lm.to_pixel("top", x, z, size), size=size)
        assert hit[0] == "top" and abs(hit[1] - x) < 1e-9 and abs(hit[2] - z) < 1e-9
        z, y = rng.uniform(0, 240), rng.uniform(-60, 40)
        hit = lm.to_world(*lm.to_pixel("side", z, y, size), size=size)
        assert hit[0] == "side" and abs(hit[1] - z) < 1e-9 and abs(hit[2] - y) < 1e-9
    assert lm.to_world(10, 595, size=size) is None          # the HUD strip is not a panel


def test_map_draws_setpoint_where_clicks_land():
    """Render with a setpoint, then look at the pixels to_pixel names: the
    magenta marker must be there, in both panels."""
    lm = LocalizationMap()
    sp = (35.0, 12.0, 120.0)
    hud = lm.hud_lines(None, 0.0, "extra", setpoint=sp)
    img = lm.render(470, 600, None, 0.0, (), True, hud, setpoint=sp, limits=LIM)
    size = (470, 600, len(hud))
    for panel, a, b in (("top", sp[0], sp[2]), ("side", sp[2], sp[1])):
        u, v = lm.to_pixel(panel, a, b, size)
        patch = img[int(round(v)) - 12:int(round(v)) + 13,
                    int(round(u)) - 12:int(round(u)) + 13].reshape(-1, 3)
        hits = np.all(np.abs(patch.astype(int) - C_SP) < 40, axis=1).sum()
        assert hits > 20, (panel, hits)


def test_hud_line_count_is_stable():
    """The panels are sized from the HUD height; if it changed with the
    pose coming and going, a click would land in a different place."""
    lm = LocalizationMap()
    sp = (0, 0, 70)
    a = lm.hud_lines(None, 0.0, "x", setpoint=sp)
    b = lm.hud_lines((1.0, 2.0, 100.0), 5.0, "x", setpoint=sp)
    assert len(a) == len(b)


# ----------------------------------------------------------------------
# mission
# ----------------------------------------------------------------------

def test_mission_holds_at_setpoint():
    from test_landing_control import World, fly
    sp = (30.0, 8.0, 110.0)
    m = Mission(MissionConfig(settle_s=0.4, hold=True, hold_point=sp, mission_max_s=80))
    w = World(p=(-20.0, 30.0, 220.0), heading=5.0)
    cmds = fly(m, w, steps=900)
    states = [s for s, _ in cmds]
    assert State.HOLD in states and State.CLOSE not in states, set(states)
    assert all(c.action != "hop" for _, c in cmds)
    assert np.linalg.norm(w.p - np.array(sp)) < 3.0, w.p
    # and facing the board from there
    face = math.degrees(math.atan2(-sp[0], sp[2]))
    assert abs(w.heading - face) < 3.0, (w.heading, face)


# ----------------------------------------------------------------------
# closed loop: the simulated Tello flies to an off-axis, raised setpoint
# ----------------------------------------------------------------------

def test_sim_flies_to_setpoint():
    """Off to the side, raised, and with the airframe sliding steadily one
    way (2.5 cm/s, a mis-trimmed Tello) on top of a little random drift:
    the integral has to take the slide out, or it holds ~5 cm off."""
    import tello_aruco_landing as tal
    from tello_pose import BoardGeometry
    from tello_sim import LandingScene, SimParams, SimTello, VirtualClock, body_axes

    sp = np.array([40.0, 12.0, 115.0])
    clock = VirtualClock()
    params = SimParams(drift_cms=1.0, bias_cms=(2.0, 0.0, -1.5))
    drone = SimTello(LandingScene(BoardGeometry(), seed=3), params, clock,
                     seed=3, start=(-0.3, 2.4, 8.0))
    drone.connect()
    drone.streamon()
    args = tal.build_parser().parse_args(
        ["--sim", "--setpoint", *(str(v) for v in sp), "--max-time", "50"])
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = tal.Flight(args, drone=drone, clock=clock).run()
    assert "HOLD" in res["states"], res
    # truth over the 4 s before it started to come down (mission timeout)
    t_land = next(t for t, e in drone.events if e.startswith("landed"))
    pts = [(t, np.array([x, y, z]), h) for t, x, y, z, h in drone.track
           if t_land - 6.0 < t < t_land - 2.0]
    cams = []
    for t, pos, hdg in pts:
        f, _, _ = body_axes(math.radians(hdg))
        cams.append((pos + f * drone.p.cam_fwd_m) * 100.0)
    cam = np.mean(cams, axis=0)
    err = np.abs(cam - sp)
    assert np.all(err < [3.5, 3.5, 3.5]), (cam, sp)
    face = math.degrees(math.atan2(-sp[0], sp[2]))
    assert abs(pts[-1][2] - face) < 5.0, (pts[-1][2], face)


# ----------------------------------------------------------------------

def main():
    sys.path.insert(0, str(ROOT / "tests"))
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print("\nPASS" if not failed else f"\nFAIL ({failed} of {len(tests)})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
