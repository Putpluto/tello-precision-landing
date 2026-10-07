"""A single marker as the landing target (board_config.py single ...):
geometry, board.json loading, the approach scaled to the marker, the
tracker's 'agree' case, landing with the side of the axis unknown, and the
printed sheet.

    python tests/test_single_marker.py
"""

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from board_config import load_board                                 # noqa: E402
from landing_control import (Mission, MissionConfig, State,          # noqa: E402
                             for_board)
from tello_pose import BoardGeometry, BoardTracker                  # noqa: E402

SINGLE = BoardGeometry(ids=(0,), marker_m=0.070)


def test_single_geometry():
    assert SINGLE.single and SINGLE.layout == {0: (0.0, 0.0)}
    pts = SINGLE.object_points()[0]
    assert np.allclose(pts.mean(axis=0), 0.0)                     # centred on the origin
    assert np.isclose(np.ptp(pts[:, 0]), 0.070)
    assert np.isclose(SINGLE.span_m, 0.070)
    assert "single marker id 0" in SINGLE.check()
    assert BoardGeometry().layout[0] == (-0.0475, 0.0475)          # the 2x2 board unchanged


def test_load_board_from_json():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "b.json"
        p.write_text(json.dumps({"ids": [0], "marker_mm": 70, "pad_drop_mm": 380,
                                 "pad_out_mm": 240, "tilt_deg": 5}))
        g = load_board(p)
        assert g.ids == (0,) and math.isclose(g.marker_m, 0.07)
        assert math.isclose(g.pad_drop_m, 0.38) and math.isclose(g.pad_out_m, 0.24)
        assert g.tilt_deg == 5.0
    assert load_board().ids == (0, 1, 2, 3)                        # TELLO_BOARD=default


def test_board_config_cli_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        f = str(pathlib.Path(d) / "board.json")
        r = subprocess.run([sys.executable, str(ROOT / "board_config.py"), "--file", f,
                            "single", "--id", "0", "--marker-mm", "70"],
                           capture_output=True, text=True, cwd=d)
        assert r.returncode == 0, r.stderr
        assert "FAR 100 cm" in r.stdout, r.stdout
        assert load_board(f) == SINGLE
        r = subprocess.run([sys.executable, str(ROOT / "board_config.py"), "--file", f,
                            "board"], capture_output=True, text=True, cwd=d)
        assert load_board(f) == BoardGeometry(), r.stdout


def test_approach_scales_with_the_target():
    cfg, lim = for_board(BoardGeometry())
    assert cfg.standoff_far == 150.0 and lim.z_max == 250.0        # the board: unchanged
    cfg, lim = for_board(SINGLE)
    assert cfg.standoff_far == 100.0 and cfg.standoff_near == 70.0
    assert lim.z_max < 250.0
    cfg, lim = for_board(BoardGeometry(ids=(0,), marker_m=0.20))  # bigger than the board
    assert cfg.standoff_far == 150.0 and lim.z_max == 250.0


def test_tracker_agree_near_the_axis():
    from test_landing_control import FakeEstimator, mirror_pair
    est = FakeEstimator()
    tr = BoardTracker(est)
    a, b = mirror_pair([1.0, 0.5, 80.0], 0.0, 0.0, ratio=1.0)     # 2.2 cm apart
    est.next = [a, b]
    out = tr.update(t_capture=0.0)
    assert out.resolved and out.source == "agree"
    a, b = mirror_pair([10.0, 0.0, 80.0], 0.0, 0.0, ratio=1.0)    # 20 cm apart
    est.next = [a, b]
    out = tr.update(t_capture=0.1)
    assert not out.resolved and out.source == "guess"


def test_lands_with_side_unknown():
    """Unresolved the whole way, 2 cm off the axis: the distance is known
    (either candidate), so it settles; the side is not, so it hops
    straight - never more than those 2 cm off."""
    from test_landing_control import World, fly
    m = Mission(MissionConfig(settle_s=0.4))
    w = World(p=(2.0, 5.0, 180.0), heading=-0.6)
    cmds = fly(m, w, steps=900, resolved=False)
    last = cmds[-1][1]
    assert last.action == "hop", (m.state, m.reason)
    assert last.hop[1] == 0 and "side unknown" in last.note, last
    assert all(c.rc[0] == 0 for _, c in cmds)                      # never steered sideways


def test_printed_single_marker():
    """make_targets.py prints the configured single marker at its size,
    centred on A4 - and the detector finds id 0 there."""
    import cv2
    with tempfile.TemporaryDirectory() as d:
        cfg = pathlib.Path(d) / "board.json"
        cfg.write_text(json.dumps({"ids": [0], "marker_mm": 150}))
        env = dict(os.environ, TELLO_BOARD=str(cfg))
        r = subprocess.run([sys.executable, str(ROOT / "make_targets.py"), "--out-dir", d,
                            "--room-markers", "0"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        pdf = pathlib.Path(d) / "landing_marker_id00_150mm_A4.pdf"
        assert pdf.exists(), list(pathlib.Path(d).iterdir())
        try:
            import pymupdf
        except ImportError:
            return                                               # rasteriser optional
        pix = pymupdf.open(str(pdf))[0].get_pixmap(dpi=100)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        corners, ids, _ = det.detectMarkers(gray)
        assert ids is not None and list(ids.flatten()) == [0], ids
        side_mm = np.linalg.norm(corners[0][0][0] - corners[0][0][1]) / 100 * 25.4
        assert abs(side_mm - 150.0) < 2.0, side_mm


def main():
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print("\nPASS" if not failed else f"\nFAIL ({failed})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
