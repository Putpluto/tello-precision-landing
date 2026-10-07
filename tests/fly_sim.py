"""
Watch the mission fly on the synthetic Tello (tests/sim.py): the camera +
map view the real mission shows, beside a 3D view of the simulated room
with the drone's true path.

    python tests/fly_sim.py                          # a live window, real time
    python tests/fly_sim.py --record sim.mp4         # ...and save it
    python tests/fly_sim.py --fast --record sim.mp4  # no window, as fast as it computes
    python tests/fly_sim.py --turns R L R R L        # a wrong turn costs time, not the mission

q in the window stops the mission (it lands). The room is sim.route_scene:
from each stop the next target is a quarter turn away. The simulated drone
is not a model of the Tello's dynamics - this shows the mission's logic
and the localization working, not how a real flight will go.
"""

import argparse
import os
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
os.environ.setdefault("TELLO_BOARD", "default")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import sim  # noqa: E402
from landing_control import BOARD, target_name  # noqa: E402
from room_map import MarkerPose, RoomMap  # noqa: E402
from room_view import RoomView3D  # noqa: E402
from tello_aruco_landing import WAYPOINTS, Lander, parse_turns  # noqa: E402
from tello_pose import NOMINAL_DIST, NOMINAL_K, BoardGeometry  # noqa: E402

FPS = 15                    # a whole number: the H.264 encoder refuses 15.000000000000002
FRAME_S = 1.0 / FPS
ROOM_W, H = 600, 720
TITLE = "simulated mission - q lands"


def room_of(sheets):
    """The simulated room as a RoomMap, so room_view.py can draw it."""
    markers = {m: MarkerPose(m, 0.15, s.axes, s.centre / 100.0)
               for m, s in sheets.items() if m != BOARD}
    b = sheets[BOARD]
    return RoomMap(markers, board_R=b.axes, board_t=b.centre / 100.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", nargs="*", default=["R", "R", "R", "R", "L"], metavar="R|L",
                    help="search turns, one per marker then the board (default: the "
                         "ones this room needs, R R R R L)")
    ap.add_argument("--record", metavar="FILE.mp4", help="save what the window shows")
    ap.add_argument("--log", metavar="FILE.csv", help="the mission's log, as on the drone")
    ap.add_argument("--fast", action="store_true",
                    help="no window, no real-time pacing (with --record)")
    args = ap.parse_args()

    geom = BoardGeometry()
    sheets, pad = sim.route_scene(geom)
    drone = sim.FakeTello(sim.Scene(sheets.values()))
    room = room_of(sheets)
    view3d = RoomView3D(geom)
    view3d.frame_points([s.centre / 100.0 for s in sheets.values()] + [pad / 100.0, [0, 0, 0]])
    view3d.az, view3d.el = np.radians(205.0), np.radians(48.0)
    stop = threading.Event()
    trail, writer, lander = [], None, None
    nxt = [0.0]
    real0 = time.monotonic()

    def emit():
        """One frame: the mission's own view, and the room from outside."""
        nonlocal writer
        b = drone.body
        trail.append(b.pos.copy())
        hud = [f"sim t {drone.t:5.1f} s   {lander.state}   {target_name(lander.target)}"]
        if drone.landed_at is not None:
            e = drone.landed_at - pad
            hud.append(f"landed {np.hypot(e[0], e[2]):.1f} cm from the pad centre")
        else:
            hud.append(f"true position x {b.pos[0]:+.0f}  y {b.pos[1]:.0f}  "
                       f"z {b.pos[2]:+.0f} cm")
        img = np.hstack([lander.render(),
                         view3d.render(ROOM_W, H, room, p_world_cm=b.pos,
                                       R_world_cam=sim.camera_axes(b.yaw),
                                       trail=trail[-2000:], hud=hud)])
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"avc1"),
                                         FPS, (img.shape[1], img.shape[0]))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             FPS, (img.shape[1], img.shape[0]))
            writer.write(img)
        if not args.fast:
            cv2.imshow(TITLE, img)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                stop.set()
            lag = drone.t - (time.monotonic() - real0)
            if lag > 0:
                time.sleep(lag)

    tick = drone.sleep

    def sleep(dt):
        """The drone's clock, with a frame drawn every FRAME_S of it - also
        through the takeoff, the hop and the landing, when the mission's
        loop is waiting on the drone."""
        end = drone.t + dt
        while drone.t < end - 1e-9:
            h = min(end, nxt[0]) - drone.t
            if h > 1e-9:
                tick(h)
            if drone.t >= nxt[0] - 1e-9:
                emit()
                nxt[0] += FRAME_S

    drone.sleep = sleep
    lander = Lander(drone, NOMINAL_K, NOMINAL_DIST, waypoints=WAYPOINTS,
                    search_turn=parse_turns(args.turns, WAYPOINTS), board=geom,
                    stop=stop, now=drone.now, sleep=sleep, log=args.log)
    reason = lander.run()
    if drone.landed_at is not None:
        e = drone.landed_at - pad
        print(f"\n{reason}: {np.hypot(e[0], e[2]):.1f} cm from the pad centre, "
              f"{drone.t:.0f} s simulated")
    if writer is not None:
        writer.release()
        print(f"recorded {args.record}")
    if not args.fast:
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
