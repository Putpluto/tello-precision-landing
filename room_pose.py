"""
Room-frame localization against a RoomMap built by map_room.py.

Generalizes tello_pose.PoseEstimator's single-board solve to whichever
subset of the room's mapped markers are visible in a frame. Pure
perception, like tello_pose.py: image + RoomMap in, camera pose in the
room's world frame out. Not wired into flight control: a visualization
and debugging tool first.

WHY TWO SOLVERS
---------------
The landing board's 4 markers all sit on one flat face, so combining
them into a single planar PnP solve is exact. Room markers are mounted
on different walls in general, so the combined point set is only
sometimes coplanar (e.g. two markers sharing one wall) and sometimes not
(e.g. a marker on each of two walls at a corner). A genuinely planar
target has the classic two-fold pose ambiguity; a non-planar one
generically doesn't. So each frame is routed to whichever solver matches
what it actually saw - see tello_pose.solve_planar_pnp /
solve_general_pnp / is_coplanar.

    python room_pose.py --live --map room_map.npz
    python room_pose.py --video walk.mp4 --map room_map.npz
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from room_map import RoomMap
from tello_io import put_text
from tello_pose import (BoardGeometry, detector_params, is_coplanar,
                        load_calibration, rotation_to_ypr, solve_general_pnp,
                        solve_planar_pnp)


@dataclass
class RoomPose:
    p_world_cm: np.ndarray      # camera position in world (room) frame, cm
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    R_world_cam: np.ndarray     # camera axes in world frame - for the 3D viewer
    reproj_rms_px: float
    n_markers: int
    ids: tuple
    t_capture: float

    def __str__(self):
        x, y, z = self.p_world_cm
        return (f"world x{x:+7.1f} y{y:+7.1f} z{z:7.1f} cm | "
                f"yaw{self.yaw_deg:+6.1f} pitch{self.pitch_deg:+6.1f} | "
                f"rms{self.reproj_rms_px:5.2f}px | n{self.n_markers} {self.ids}")


class RoomLocalizer:
    def __init__(self, K, dist, room_map: RoomMap, dict_id=cv2.aruco.DICT_4X4_50,
                 max_rms_px=3.0, long_range=True, geom=BoardGeometry()):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64)
        self.max_rms = max_rms_px
        self.geom = geom
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector = cv2.aruco.ArucoDetector(d, detector_params(long_range))
        self.set_map(room_map)

    def set_map(self, room_map):
        """Swap the map without rebuilding the detector - the GUI re-solves
        a provisional map while you walk, and constructing an ArucoDetector
        every second to follow it would be pure waste."""
        self.map = room_map
        # If the map knows where the board is, its markers are just four
        # more known points in the room - which is what keeps the room
        # solve alive close to the pad, where the wall markers have
        # dropped out of frame behind the drone.
        self.board_obj_world = {}
        if room_map is not None and room_map.has_board:
            for mid, pts in self.geom.object_points().items():
                self.board_obj_world[mid] = (
                    (room_map.board_R @ pts.T.astype(np.float64)).T
                    + room_map.board_t)

    # -- detection ----------------------------------------------------
    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMarkers(gray)

    def _gather(self, corners, ids):
        """Keep only markers this map knows about - an unmapped marker in
        frame is silently ignored rather than poisoning the solve. The
        landing board's markers count only once the map has placed it."""
        obj, img, used = [], [], []
        if ids is None:
            return None, None, ()
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid in self.map.markers:
                obj.append(self.map.markers[mid].object_points_world())
            elif mid in self.board_obj_world:
                obj.append(self.board_obj_world[mid])
            else:
                continue
            img.append(c.reshape(-1, 2))
            used.append(mid)
        if not obj:
            return None, None, ()
        return (np.concatenate(obj).astype(np.float32),
                np.concatenate(img).astype(np.float32),
                tuple(sorted(used)))

    # -- pose -----------------------------------------------------------
    def estimate(self, frame, t_capture=None, detection=None):
        """Return a RoomPose, or None if no mapped marker was resolved.

        detection: an optional (corners, ids) from a detect() the caller
        already paid for. tello_aruco_landing.py runs the board estimator
        and this one off a single detection pass per frame.
        """
        t_capture = time.time() if t_capture is None else t_capture
        corners, ids = (detection if detection is not None
                        else self.detect(frame)[:2])
        obj, img, used = self._gather(corners, ids)
        if obj is None or len(obj) < 4:
            return None            # a single marker's 4 points is the floor

        solve = solve_planar_pnp if is_coplanar(obj) else solve_general_pnp
        rv, tv, rms = solve(obj, img, self.K, self.dist)
        if rv is None or rms > self.max_rms:
            return None

        R_wc, _ = cv2.Rodrigues(rv)          # world -> camera
        R_cw = R_wc.T                        # camera axes in world frame
        p = (-R_wc.T @ tv).ravel()           # camera position in world frame, m

        yaw, pitch, roll = rotation_to_ypr(R_cw)

        return RoomPose(
            p_world_cm=p * 100.0,
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            R_world_cam=R_cw,
            reproj_rms_px=rms, n_markers=len(used), ids=used,
            t_capture=t_capture,
        )

    # -- overlay ----------------------------------------------------------
    def draw(self, frame, pose, corners=None, ids=None):
        out = frame.copy()
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        if pose is None:
            cv2.putText(out, "NOT LOCALIZED", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return out
        x, y, z = pose.p_world_cm
        lines = [
            f"world  x {x:+7.1f}  y {y:+7.1f}  z {z:7.1f} cm",
            f"yaw {pose.yaw_deg:+6.1f}  pitch {pose.pitch_deg:+6.1f}  "
            f"roll {pose.roll_deg:+6.1f} deg",
            f"rms {pose.reproj_rms_px:.2f} px   n {pose.n_markers}   "
            f"ids {pose.ids}",
        ]
        for i, s in enumerate(lines):
            put_text(out, s, (12, 30 + 26 * i), 0.62, (120, 200, 255))
        return out


# ----------------------------------------------------------------------
# Standalone diagnostics (mirrors tello_pose.py's --live/--video)
# ----------------------------------------------------------------------

def main():
    from map_room import drone_frames
    from tello_io import file_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--map", default="room_map.npz")
    ap.add_argument("--calib", default="tello_calib.npz")
    args = ap.parse_args()

    if not any([args.live, args.video, args.image]):
        ap.error("pick one of --live, --video, --image")

    room_map = RoomMap.load(args.map)
    print(f"loaded {room_map}")
    K, dist, real = load_calibration(args.calib)
    from board_config import load_board
    geom = load_board()
    est = RoomLocalizer(K, dist, room_map, geom=geom, dict_id=geom.dict_id)

    if args.image:
        img = cv2.imread(args.image)
        pose = est.estimate(img)
        corners, ids, _ = est.detect(img)
        print(pose if pose else "not localized")
        cv2.imshow("room pose", est.draw(img, pose, corners, ids))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    frames = drone_frames() if args.live else file_frames(args.video)
    print("keys: q quit")
    for frame in frames:
        if frame is None:
            continue
        corners, ids, _ = est.detect(frame)
        pose = est.estimate(frame)
        cv2.imshow("room pose", est.draw(frame, pose, corners, ids))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
