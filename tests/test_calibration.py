"""End-to-end check of calibrate_camera.py against a camera whose
intrinsics are known exactly: the simulator renders the printed chessboard
from a varied set of views (the way the README says to collect them) and
from a degenerate set (the way the shipped calib/ frames were collected),
and the calibrator has to recover the first and refuse the second.

    python tests/test_calibration.py
"""

import math
import os
import pathlib
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
import calibrate_camera as cc                                     # noqa: E402
from tello_pose import BoardGeometry                               # noqa: E402
from tello_sim import (SIM_DIST, SIM_K, LandingScene, LensEffects,  # noqa: E402
                       lookat, rot_y)

CHESS_R = rot_y(0.0)
CHESS_T = np.array([0.0, 0.0, -0.9])      # on the wall behind the board, facing the room


def render_set(views, seed=0):
    scene = LandingScene(BoardGeometry(), chessboard=(CHESS_R, CHESS_T), pad=False)
    # take the landing board out: this is a chessboard session
    scene.items = [it for it in scene.items if it.name not in ("board", "stand")]
    fx = LensEffects(SIM_K, SIM_DIST, (960, 720), blur_sigma=0.6, noise_sigma=2.0,
                     jpeg_quality=85, seed=seed)
    imgs = []
    for p, aim in views:
        R_cw, t_cw = lookat(CHESS_T + p, CHESS_T + aim)
        imgs.append(fx.apply(scene.render(SIM_K, R_cw, t_cw)))
    return imgs


def good_views(n=24):
    """Tilted 15-45 deg in all directions, 0.35-0.8 m, board all over the
    frame - the recipe in calibrate_camera.py's docstring."""
    rng = np.random.default_rng(4)
    out = []
    for i in range(n):
        r = rng.uniform(0.38, 0.8)
        tilt = math.radians(rng.uniform(15, 45))
        az = rng.uniform(0, 2 * math.pi)
        p = r * np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az),
                          math.cos(tilt)])
        aim = np.array([rng.uniform(-0.09, 0.09), rng.uniform(-0.07, 0.07), 0.0])
        out.append((p, aim))
    return out


def degenerate_views(n=14):
    """What calib/ contains: same distance, same tilt, slid sideways."""
    out = []
    for i in range(n):
        x = -0.25 + 0.5 * i / (n - 1)
        out.append((np.array([x, 0.55, 1.35]), np.array([x, 0.0, 0.0])))
    return out


def run(views, seed):
    with tempfile.TemporaryDirectory() as d:
        files = []
        for i, img in enumerate(render_set(views, seed)):
            fn = str(pathlib.Path(d) / f"calib_{i:03d}.png")
            cv2.imwrite(fn, img)
            files.append(fn)
        return cc.calibrate(files)


def test_good_set_recovers_intrinsics():
    r = run(good_views(), 1)
    fx, fy = r["K"][0, 0], r["K"][1, 1]
    assert abs(fx / SIM_K[0, 0] - 1) < 0.015, (fx, r["problems"])
    assert abs(r["K"][0, 2] - SIM_K[0, 2]) < 15 and abs(r["K"][1, 2] - SIM_K[1, 2]) < 15
    assert not r["problems"], r["problems"]
    return r


def test_degenerate_set_is_refused():
    r = run(degenerate_views(), 2)
    assert r["problems"], "a set of near-identical views must not pass"
    return r


def main():
    ok = True
    for name, fn in (("varied views", test_good_set_recovers_intrinsics),
                     ("degenerate views", test_degenerate_set_is_refused)):
        try:
            r = fn()
            print(f"  ok    {name}: fx {r['K'][0,0]:.1f} +- {r['std'][0]:.1f} "
                  f"(true {SIM_K[0,0]:.0f}), rms {r['rms']:.3f}, "
                  f"{len(r['problems'])} problem(s)")
            for p in r["problems"]:
                print(f"          - {p}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
    print("\nPASS" if ok else "\nFAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
