"""Pose estimation on rendered images, and the Kalman filter's gating."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import sim
from tello_pose import (NOMINAL_DIST, NOMINAL_K, BoardGeometry, BoardTracker, PoseEstimator,
                        PoseFilter)

GEOM = BoardGeometry()


def view(sheet, pos, yaw):
    scene = sim.Scene([sheet])
    return cv2.cvtColor(scene.render(pos, yaw), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("pos, yaw", [
    ((0, 0, 150), 180),          # square on at FAR
    ((30, 15, 100), 170),        # off to the right, above, turned toward it
    ((-20, 0, 70), 190),         # NEAR, to the left
])
def test_board_pose_matches_the_geometry(pos, yaw):
    board = sim.Sheet(sim.board_art(GEOM), (0, 0, 0), (0, 0, 1))
    est = PoseEstimator(NOMINAL_K, NOMINAL_DIST, GEOM)
    p = BoardTracker(est).update(view(board, pos, yaw), t_capture=0.0)
    truth, R = board.to_local(pos, yaw)
    assert p is not None and p.resolved
    assert np.abs(p.p_board_cm - truth).max() < 1.5
    assert np.abs(p.R_level_cam - R).max() < 0.03


def test_waypoint_marker_pose():
    marker = sim.Sheet(sim.marker_art(GEOM.dict_id, 6), (0, 0, 0), (0, 0, 1))
    geom = BoardGeometry(dict_id=GEOM.dict_id, marker_m=0.15, ids=(6,))
    est = PoseEstimator(NOMINAL_K, NOMINAL_DIST, geom)
    p = BoardTracker(est).update(view(marker, (-25, 5, 100), 190), t_capture=0.0)
    assert np.abs(p.p_board_cm - [-25, 5, 100]).max() < 1.5


def test_upside_down_board_is_refused_and_reported():
    board = sim.Sheet(cv2.rotate(sim.board_art(GEOM), cv2.ROTATE_180), (0, 0, 0), (0, 0, 1))
    est = PoseEstimator(NOMINAL_K, NOMINAL_DIST, GEOM)
    frame = view(board, (0, 0, 120), 180)
    det = est.detect(frame)[:2]
    assert est.estimate(detection=det) is None
    assert est.upside_down(det)


def test_filter_follows_a_real_jump_instead_of_latching():
    """The old filter froze on its last value after a 40 cm jump; this one
    restarts on the measurement after a few consecutive rejections."""
    f = PoseFilter()

    def meas(x, t):
        return SimpleNamespace(p_board_cm=np.array([x, 0.0, 120.0]), t_capture=t,
                               resolved=True, degraded=False)

    t = 0.0
    for _ in range(10):
        f.update(meas(0.0, t), now=t)
        t += 1 / 15
    for _ in range(f.reinit_after + 1):
        f.update(meas(60.0, t), now=t)
        t += 1 / 15
    assert f.position()[0] == pytest.approx(60.0, abs=2.0)
