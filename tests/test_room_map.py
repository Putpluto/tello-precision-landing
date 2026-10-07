"""Analytic check of map_room.py / room_pose.py: no images, no drone,
no PDFs - just known marker world poses, projected corners, and a
comparison against ground truth. Mirrors test_pose_roundtrip.py's
analytic_check(), extended to a multi-wall room instead of one flat
board, which is what actually exercises the two new pieces this repo
didn't have before: multi-marker graph chaining (map_room.py) and the
coplanar-vs-general PnP branch (room_pose.py, tello_pose.is_coplanar).
"""

import os
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
from map_room import _compose_world, _relative_transform          # noqa
from room_map import MarkerPose, RoomMap                          # noqa
from room_pose import RoomLocalizer                                # noqa
from tello_pose import is_coplanar, solve_general_pnp, solve_planar_pnp  # noqa

F = 684.0
K = np.array([[F, 0, 480.0], [0, F, 360.0], [0, 0, 1.0]])
DIST = np.zeros(5)
SIZE_M = 0.15


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def lookat(p_cam, aim):
    """world->camera (R_wc, t_wc) - same convention used throughout the repo
    (tests/test_pose_roundtrip.py, tello_gui.py)."""
    p = np.asarray(p_cam, float)
    z_c = np.asarray(aim, float) - p
    z_c = z_c / np.linalg.norm(z_c)
    up = np.array([0.0, 1.0, 0.0])
    x_c = np.cross(-up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R_wc = np.column_stack([x_c, y_c, z_c]).T
    return R_wc, (-R_wc @ p).reshape(3)


def marker_cam_pose(R_wc, t_wc, R_m, t_m):
    """Ground-truth marker->camera pose - what solve_planar_pnp would
    recover from noiseless corners, computed directly instead."""
    return R_wc @ R_m, R_wc @ t_m + t_wc


# A corner: two markers (4, 5) on one wall (world z=0 plane), one marker
# (6) on a perpendicular wall - the minimum case that has both a coplanar
# pair and a non-planar pair.
GT = {
    4: (np.eye(3), np.array([0.0, 0.0, 0.0])),
    5: (np.eye(3), np.array([1.0, 0.0, 0.0])),
    6: (rot_y(90.0), np.array([0.5, 0.0, -1.0])),
}


def check_graph_chaining():
    """map_room.py's BFS composition, isolated from detection/PnP: feed it
    hand-computed ground-truth marker->camera observations directly and
    check it recovers the true marker world poses."""
    cam_a, aim_a = np.array([0.4, -0.1, 1.4]), (0.5, 0.0, 0.0)
    cam_b, aim_b = np.array([1.8, 0.1, -1.6]), (0.75, 0.0, -0.5)
    R_wc_a, t_wc_a = lookat(cam_a, aim_a)
    R_wc_b, t_wc_b = lookat(cam_b, aim_b)

    obs_a = {i: marker_cam_pose(R_wc_a, t_wc_a, *GT[i]) for i in (4, 5)}
    obs_b = {i: marker_cam_pose(R_wc_b, t_wc_b, *GT[i]) for i in (5, 6)}
    R_45, t_45 = _relative_transform(*obs_a[4], *obs_a[5])
    R_56, t_56 = _relative_transform(*obs_b[5], *obs_b[6])

    world = {4: (np.eye(3), np.zeros(3))}
    world[5] = _compose_world(*world[4], R_45, t_45)
    world[6] = _compose_world(*world[5], R_56, t_56)

    worst = 0.0
    for i, (R_gt, t_gt) in GT.items():
        R_rec, t_rec = world[i]
        worst = max(worst, float(np.linalg.norm(t_rec - t_gt)) * 100.0)
        dR = R_rec.T @ R_gt
        worst = max(worst, float(np.degrees(np.arccos(
            np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
    return worst


def check_localizer():
    """room_pose.RoomLocalizer against rendered (projected) corners for a
    coplanar pair, a non-planar pair, and all three markers together -
    the three branches is_coplanar can route between."""
    room_map = RoomMap({i: MarkerPose(i, SIZE_M, R, t)
                        for i, (R, t) in GT.items()})

    def project(mid, R_wc, t_wc):
        corners = room_map.markers[mid].object_points_world().astype(np.float32)
        rvec, _ = cv2.Rodrigues(R_wc)
        img, _ = cv2.projectPoints(corners, rvec, t_wc, K, DIST)
        return img.reshape(-1, 2)

    cases = [
        (np.array([0.4, -0.1, 1.4]), (0.5, 0.0, 0.0), (4, 5), True),
        (np.array([1.8, 0.1, -1.6]), (0.75, 0.0, -0.5), (5, 6), False),
        (np.array([0.6, -0.15, 0.6]), (0.5, 0.0, -0.3), (4, 5, 6), False),
    ]
    worst = 0.0
    for cam, aim, ids, expect_coplanar in cases:
        R_wc, t_wc = lookat(cam, aim)
        obj = np.concatenate([room_map.markers[i].object_points_world()
                              for i in ids]).astype(np.float32)
        img = np.concatenate([project(i, R_wc, t_wc)
                              for i in ids]).astype(np.float32)
        coplanar = is_coplanar(obj)
        assert coplanar == expect_coplanar, (
            f"is_coplanar({ids}) = {coplanar}, expected {expect_coplanar}")
        solve = solve_planar_pnp if coplanar else solve_general_pnp
        rv, tv, rms = solve(obj, img, K, DIST)
        R_wc_est, _ = cv2.Rodrigues(rv)
        p_est = (-R_wc_est.T @ tv).ravel()
        worst = max(worst, float(np.linalg.norm(p_est - cam)) * 100.0)
    return worst


BOARD_GT = (rot_y(-30.0), np.array([0.2, 0.0, -1.6]))


def check_board_chaining():
    """The landing board as one more node in the graph, plus the
    board<->room transform the coarse leg flies on. This is the bridge
    between the room map and tello_aruco_landing.py: if it is wrong the
    drone flies confidently to the wrong place."""
    R_b_gt, t_b_gt = BOARD_GT
    # a frame that sees marker 5 and the board together
    R_wc, t_wc = lookat(np.array([1.6, 0.1, -0.9]), (0.6, 0.0, -1.2))
    obs_5 = marker_cam_pose(R_wc, t_wc, *GT[5])
    obs_b = marker_cam_pose(R_wc, t_wc, R_b_gt, t_b_gt)
    R_5b, t_5b = _relative_transform(*obs_5, *obs_b)
    R_rec, t_rec = _compose_world(*GT[5], R_5b, t_5b)

    worst = float(np.linalg.norm(t_rec - t_b_gt)) * 100.0
    dR = R_rec.T @ R_b_gt
    worst = max(worst, float(np.degrees(np.arccos(
        np.clip((np.trace(dR) - 1) / 2, -1, 1)))))

    # the transform itself: a standoff point 150 cm out along the board's
    # own +Z must land 150 cm out along the board normal in room frame
    rm = RoomMap({i: MarkerPose(i, SIZE_M, R, t) for i, (R, t) in GT.items()},
                 R_rec, t_rec)
    p_board = np.array([0.0, 0.0, 150.0])
    p_world = rm.board_to_world_cm(p_board)
    expect = (R_b_gt @ np.array([0.0, 0.0, 1.5]) + t_b_gt) * 100.0
    worst = max(worst, float(np.linalg.norm(p_world - expect)))
    worst = max(worst, float(np.linalg.norm(
        rm.world_to_board_cm(p_world) - p_board)))
    return worst


def check_coarse_signs():
    """Sign table for the coarse leg in the room frame - the same
    landing_control.ServoLaw the board-relative phases use, here fed room
    coordinates. Drone at (0,0,150) cm facing -Z, i.e. square on to a
    board at the room origin."""
    from landing_control import ServoLaw

    # camera +X = room +X (right), +Y = room -Y (down), +Z = room -Z (fwd)
    R_world_cam = np.array([[1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0]])
    p = np.array([0.0, 0.0, 150.0])
    origin = np.array([0.0, 0.0, 0.0])

    def one(target, aim=origin):
        out = ServoLaw().step(p, None, R_world_cam, np.array(target), aim)
        return (*out.rc, out.dist)

    a, b, c, d, dist = one([50.0, 0.0, 150.0])           # to its right
    cases = [("target right -> a > 0 (rc right)", a > 0)]
    a, b, c, d, _ = one([-50.0, 0.0, 150.0])             # to its left
    cases.append(("target left -> a < 0", a < 0))
    a, b, c, d, _ = one([0.0, 0.0, 100.0])               # nearer the board
    cases.append(("target ahead -> b > 0 (rc forward)", b > 0))
    a, b, c, d, _ = one([0.0, 0.0, 200.0])               # further away
    cases.append(("target behind -> b < 0", b < 0))
    a, b, c, d, _ = one([0.0, 50.0, 150.0])              # above
    cases.append(("target above -> c > 0 (rc climb)", c > 0))
    a, b, c, d, _ = one([0.0, -50.0, 150.0])             # below
    cases.append(("target below -> c < 0", c < 0))
    # board off to the right -> yaw clockwise to face it
    a, b, c, d, _ = one([0.0, 0.0, 150.0], aim=np.array([100.0, 0.0, 0.0]))
    cases.append(("board right -> d > 0 (yaw CW)", d > 0))
    a, b, c, d, _ = one([0.0, 0.0, 150.0], aim=np.array([-100.0, 0.0, 0.0]))
    cases.append(("board left -> d < 0", d < 0))

    bad = [name for name, ok in cases if not ok]
    for name, ok in cases:
        print(f"      {'ok ' if ok else 'BAD'}  {name}")
    return bad


def check_save_load(tmp_path):
    room_map = RoomMap({i: MarkerPose(i, SIZE_M, R, t)
                        for i, (R, t) in GT.items()}, *BOARD_GT)
    room_map.save(str(tmp_path))
    loaded = RoomMap.load(str(tmp_path))
    assert loaded.has_board, "board pose did not survive the npz round-trip"
    worst = max(float(np.linalg.norm(loaded.markers[i].t - room_map.markers[i].t))
                for i in GT)
    return max(worst, float(np.linalg.norm(loaded.board_t - room_map.board_t)))


def main():
    import tempfile
    a = check_graph_chaining()
    b = check_localizer()
    d_board = check_board_chaining()
    with tempfile.TemporaryDirectory() as d:
        c = check_save_load(pathlib.Path(d) / "room_map_test.npz")

    print(f"[1] graph chaining   worst {a:.4f} cm/deg  (BFS pose composition)")
    print(f"[2] localizer PnP    worst {b:.4f} cm       "
          "(coplanar + non-planar branches)")
    print(f"[3] save/load        worst {c:.6f} cm       (npz round-trip)")
    print(f"[4] board bridge     worst {d_board:.4f} cm/deg  "
          "(board node + board<->room transform)")
    print("[5] coarse sign table  (room-frame control law)")
    bad = check_coarse_signs()

    ok = a < 1e-3 and b < 0.1 and c < 1e-6 and d_board < 1e-3 and not bad
    print("\nPASS" if ok else f"\nFAIL {bad}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
