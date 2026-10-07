"""
Live 3D view of the room map + drone pose, via rerun (pip install
rerun-sdk - see requirements.txt). A debugging/visualization tool: it
runs RoomLocalizer and shows where it thinks the camera is. It does not
fly the drone - see README for why this stays standalone for now.

    python view_room_3d.py --live --map room_map.npz
    python view_room_3d.py --video walk.mp4 --map room_map.npz

Opens the rerun viewer window (spawn=True). Markers are logged once as
static geometry; the camera position/orientation and its trail update
every frame on rerun's "time" timeline, so the scrubber lets you replay
a recorded run.
"""

import argparse

import numpy as np

try:
    import rerun as rr
except ImportError:                     # optional extra, see requirements.txt
    raise SystemExit("view_room_3d.py needs rerun: pip install rerun-sdk\n"
                     "(or use the GUI's Room tabs / room_view.py, which need nothing extra)")

from room_map import RoomMap
from room_pose import RoomLocalizer
from tello_pose import load_calibration

AXIS_LEN_M = 0.15
AXIS_COLORS = [[255, 60, 60], [60, 255, 60], [60, 60, 255]]     # x, y, z


def log_map(room_map: RoomMap):
    """Static geometry: logged once, visible at every point on the
    timeline regardless of where the scrubber sits."""
    for mid, m in room_map.markers.items():
        corners = m.object_points_world()
        loop = np.vstack([corners, corners[:1]])
        rr.log(f"world/markers/{mid}", rr.LineStrips3D([loop], colors=[[80, 180, 255]]),
               static=True)
        rr.log(f"world/markers/{mid}/label",
               rr.Points3D([corners.mean(axis=0)], labels=[str(mid)],
                           colors=[[80, 180, 255]], radii=0.01),
               static=True)


def log_pose(pose, trail):
    rr.set_time("time", timestamp=pose.t_capture)
    pos_m = pose.p_world_cm / 100.0
    trail.append(pos_m)

    rr.log("world/drone", rr.Points3D([pos_m], colors=[[255, 90, 90]], radii=0.04))
    vectors = [pose.R_world_cam[:, k] * AXIS_LEN_M for k in range(3)]
    rr.log("world/drone/axes", rr.Arrows3D(
        origins=[pos_m] * 3, vectors=vectors, colors=AXIS_COLORS))
    if len(trail) > 1:
        rr.log("world/drone/trail",
               rr.LineStrips3D([np.array(trail)], colors=[[255, 200, 60]]))


def main():
    from map_room import drone_frames
    from tello_io import file_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Tello stream, no takeoff")
    ap.add_argument("--video", help="pre-recorded walkthrough or flight")
    ap.add_argument("--map", default="room_map.npz")
    ap.add_argument("--calib", default="tello_calib.npz")
    args = ap.parse_args()
    if not args.live and not args.video:
        ap.error("pick one of --live, --video")

    room_map = RoomMap.load(args.map)
    print(f"loaded {room_map}")
    K, dist, real = load_calibration(args.calib)
    if not real:
        print("(nominal intrinsics - positions will be off by a few percent)")
    from board_config import load_board
    geom = load_board()
    est = RoomLocalizer(K, dist, room_map, geom=geom, dict_id=geom.dict_id)

    rr.init("tello_room", spawn=True)
    log_map(room_map)

    trail = []
    n_seen = n_total = 0
    frames = drone_frames() if args.live else file_frames(args.video)
    print("close the rerun window, or ctrl-c here, to stop")
    try:
        for frame in frames:
            if frame is None:
                continue
            n_total += 1
            pose = est.estimate(frame)
            if pose is not None:
                n_seen += 1
                log_pose(pose, trail)
    except KeyboardInterrupt:
        pass
    print(f"localized {n_seen}/{n_total} frames")


if __name__ == "__main__":
    main()
