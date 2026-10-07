"""
Build a RoomMap by walking a camera around the room - no tape measure,
no manual survey. Every marker's world pose is inferred from frames
where it shares the view with a marker whose pose is already known.

    python map_room.py --video walkthrough.mp4 --out room_map.npz
    python map_room.py --live --out room_map.npz

HOW IT WORKS
------------
1. Every frame, solve each visible marker's pose independently (a single
   ArUco marker is always planar, so solve_planar_pnp's two-fold
   disambiguation applies one marker at a time - see tello_pose.py).
2. Whenever two markers are visible in the same frame, that frame pins
   down the rigid transform between them. Keep the lowest-reprojection-
   -error observation of each pair across the whole video as that pair's
   edge (v1: no bundle adjustment / rotation averaging across repeated
   observations - if the map looks warped, that's the first thing to add).
3. Pick a root marker (most-observed by default) and breadth-first over
   the edge graph, composing transforms outward from it. The root's own
   pose becomes the identity - i.e. the room's world frame.
4. Any edge NOT used by that BFS spanning tree is a free consistency
   check: recompute it from the two markers' solved world poses and
   compare to what was actually observed. Large discrepancies mean the
   walkthrough didn't have enough overlap, or a marker moved.

WALKTHROUGH TIPS
-----------------
Move slowly. Every marker needs to share at least one frame with a
marker that is (transitively) connected to the root - so keep at least
one already-seen marker in frame while a new one comes into view, the
same way you'd pan a phone to stitch a panorama.

ID SPACE: only ids >= room_map.ROOM_ID_MIN are considered, so a landing
board (ids 0-3) accidentally in frame is ignored, not mixed into the map.
"""

import argparse
import itertools
from collections import defaultdict, deque

import cv2
import numpy as np

from room_map import BOARD_NODE, ROOM_ID_MIN, MarkerPose, RoomMap
from tello_pose import (BoardGeometry, detector_params, load_calibration,
                        solve_planar_pnp)


def _marker_object_points(size_m):
    h = size_m / 2.0
    return np.array([
        [-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0],
    ], dtype=np.float32)


def _node_name(i):
    return "board" if i == BOARD_NODE else str(i)


def _relative_transform(R_i, t_i, R_j, t_j):
    """R,t such that point_j = R @ point_i + t, for two markers' poses
    relative to the same camera in one frame (R_i/t_i, R_j/t_j both
    marker -> camera)."""
    R = R_j.T @ R_i
    t = R_j.T @ (t_i - t_j)
    return R, t


def _compose_world(R_world_i, t_world_i, R_ij, t_ij):
    """World pose of marker j, given marker i's world pose and the edge
    i->j (point_j = R_ij @ point_i + t_ij). Derivation: substitute
    point_i = R_ij.T @ (point_j - t_ij) into point_world = R_world_i @
    point_i + t_world_i."""
    R_world_j = R_world_i @ R_ij.T
    t_world_j = t_world_i - R_world_j @ t_ij
    return R_world_j, t_world_j


class RoomMapper:
    def __init__(self, K, dist, marker_size_m, dict_id=cv2.aruco.DICT_4X4_50,
                 max_rms_px=3.0, long_range=True, geom=BoardGeometry(),
                 include_board=True, min_board_markers=2):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64)
        self.marker_size_m = marker_size_m
        self.max_rms = max_rms_px
        self.obj_local = _marker_object_points(marker_size_m)
        # The board goes in as ONE node, not four: its internal layout is
        # already known, so solving it whole is both more accurate and the
        # thing the landing controller actually needs a pose for.
        self.include_board = include_board
        self.min_board_markers = min_board_markers
        self.board_obj = geom.object_points() if include_board else {}
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector = cv2.aruco.ArucoDetector(d, detector_params(long_range))
        # edges[i][j] = (R_ij, t_ij, err) ; kept symmetric, lowest-err wins
        self.edges = defaultdict(dict)
        self.seen_count = defaultdict(int)
        self.frames_used = 0

    def _solve_pts(self, obj, img):
        rv, tv, err = solve_planar_pnp(obj, img, self.K, self.dist)
        if rv is None or err > self.max_rms:
            return None
        R, _ = cv2.Rodrigues(rv)
        return R, tv.ravel(), err

    def _solve_one(self, corners):
        return self._solve_pts(self.obj_local,
                               corners.reshape(-1, 2).astype(np.float32))

    def _solve_board(self, by_id):
        """The whole board as one rigid body, in board frame. Needs at
        least min_board_markers of it - on the 2x2 board a single marker
        would pin the board's pose off one corner of it, and that error
        lands straight on the coarse waypoint the drone flies to. (A
        single-marker target is the board: min_board_markers=1.)"""
        obj, img = [], []
        for mid, pts in by_id.items():
            if mid in self.board_obj:
                obj.append(self.board_obj[mid])
                img.append(pts)
        if len(obj) < self.min_board_markers:
            return None
        return self._solve_pts(np.concatenate(obj).astype(np.float32),
                               np.concatenate(img).astype(np.float32))

    def add_frame(self, frame, detection=None):
        """detection: an optional (corners, ids) the caller already has, so
        a GUI running several estimators off one frame pays for
        detectMarkers once."""
        if detection is not None:
            corners, ids = detection
        else:
            gray = (frame if frame.ndim == 2
                    else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return 0
        poses = {}
        board_seen = {}
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            pts = c.reshape(-1, 2)
            if mid < ROOM_ID_MIN:
                if mid in self.board_obj:
                    board_seen[mid] = pts
                continue
            solved = self._solve_one(c)
            if solved is not None:
                poses[mid] = solved
                self.seen_count[mid] += 1
        if board_seen:
            solved = self._solve_board(board_seen)
            if solved is not None:
                poses[BOARD_NODE] = solved
                self.seen_count[BOARD_NODE] += 1
        if poses:
            self.frames_used += 1
        for (i, (R_i, t_i, e_i)), (j, (R_j, t_j, e_j)) in itertools.combinations(poses.items(), 2):
            R_ij, t_ij = _relative_transform(R_i, t_i, R_j, t_j)
            quality = e_i + e_j     # lower combined reprojection err = more trustworthy edge
            cur = self.edges[i].get(j)
            if cur is None or quality < cur[2]:
                self.edges[i][j] = (R_ij, t_ij, quality)
                R_ji, t_ji = _relative_transform(R_j, t_j, R_i, t_i)
                self.edges[j][i] = (R_ji, t_ji, quality)
        return len(poses)

    def build(self, root_id=None, verbose=True):
        """verbose=False is for the GUI, which re-solves a provisional map
        every second to draw it - the same solve, without the commentary."""
        if not self.edges and not self.seen_count:
            raise RuntimeError("no markers observed - nothing to map")
        if root_id is None:
            # Anchor the room to a wall marker, never the board: the board
            # is the thing most likely to be repositioned between sessions,
            # and re-siting the pad should not invalidate the whole room.
            wall = {i: n for i, n in self.seen_count.items() if i >= ROOM_ID_MIN}
            if not wall:
                raise RuntimeError("only the landing board was observed - "
                                   "no room markers to anchor the map to")
            root_id = max(wall, key=wall.get)
        elif root_id not in self.seen_count:
            raise RuntimeError(f"root id {root_id} was never observed")

        world = {root_id: (np.eye(3), np.zeros(3))}
        q = deque([root_id])
        while q:
            i = q.popleft()
            R_world_i, t_world_i = world[i]
            for j, (R_ij, t_ij, _) in self.edges[i].items():
                if j in world:
                    continue
                world[j] = _compose_world(R_world_i, t_world_i, R_ij, t_ij)
                q.append(j)

        unreached = set(self.seen_count) - set(world)
        if verbose:
            if unreached:
                print(f"!! {len(unreached)} node(s) never connected to root "
                      f"{root_id}, dropped: "
                      f"{[_node_name(i) for i in sorted(unreached)]}")
                print("   (need a frame that sees one of them together with "
                      "an already-connected marker)")
            self._report_consistency(world)

        board_R, board_t = world.pop(BOARD_NODE, (None, None))
        if verbose and self.include_board and board_R is None:
            print("!! the landing board is NOT in this map - the 3D views cannot "
                  "show the pad, and room localization cannot use the board's "
                  "markers. Re-walk with the board and a room marker in frame together.")
        markers = {i: MarkerPose(i, self.marker_size_m, R, t)
                   for i, (R, t) in world.items()}
        return RoomMap(markers, board_R, board_t), root_id

    def _report_consistency(self, world):
        """Edges not used by the BFS tree are a free check: recompute the
        pair's relative transform from the solved world poses and compare
        to what the video actually observed."""
        worst_cm, worst_deg = 0.0, 0.0
        n_checked = 0
        for i in world:
            for j, (R_ij, t_ij, _) in self.edges[i].items():
                if j <= i or j not in world:
                    continue
                R_world_i, t_world_i = world[i]
                R_world_j, t_world_j = world[j]
                R_pred, t_pred = _relative_transform(
                    R_world_i, t_world_i, R_world_j, t_world_j)
                # observed edge maps i->j in marker-local frames directly
                dt_cm = float(np.linalg.norm(t_pred - t_ij)) * 100.0
                dR = R_pred.T @ R_ij
                ang_deg = float(np.degrees(np.arccos(
                    np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))))
                worst_cm = max(worst_cm, dt_cm)
                worst_deg = max(worst_deg, ang_deg)
                n_checked += 1
        if n_checked:
            print(f"consistency check: {n_checked} redundant edge(s), "
                  f"worst {worst_cm:.1f} cm / {worst_deg:.1f} deg "
                  "(large numbers => not enough overlap, or a marker moved)")
        else:
            print("consistency check: no redundant edges (a tree, not a "
                  "graph) - can't cross-check, but the map is still usable")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def drone_frames():
    """BGR frames from a connected Tello, never taking
    off. tello_io handles the RGB and placeholder traps."""
    from tello_io import open_drone, tello_frames
    t = open_drone()
    print(f"battery {t.get_battery()}%  -- NOT taking off")
    try:
        yield from tello_frames(t)
    finally:
        t.streamoff()
        t.end()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="Tello stream, no takeoff")
    ap.add_argument("--video", help="pre-recorded walkthrough")
    ap.add_argument("--calib", default="tello_calib.npz")
    ap.add_argument("--marker-size", type=float, default=0.15,
                    help="room marker edge length, metres (default 0.15)")
    ap.add_argument("--root-id", type=int, default=None,
                    help="force this marker as the world-frame origin "
                         "(default: whichever was seen most)")
    ap.add_argument("--out", default="room_map.npz")
    ap.add_argument("--view", action="store_true", help="show detections live")
    ap.add_argument("--no-board", action="store_true",
                    help="ignore the landing board; the map will not "
                         "place the board or the pad")
    args = ap.parse_args()

    if not args.live and not args.video:
        ap.error("pick one of --live, --video")

    K, dist, real = load_calibration(args.calib)
    if not real:
        print("(nominal intrinsics - map scale will be off by the same "
              "few percent tello_pose.py warns about)")

    from board_config import load_board
    geom = load_board()
    mapper = RoomMapper(K, dist, args.marker_size, geom=geom, dict_id=geom.dict_id,
                        include_board=not args.no_board,
                        # a single-marker target is the board: one marker places it
                        min_board_markers=1 if geom.single else 2)
    from tello_io import file_frames
    frames = drone_frames() if args.live else file_frames(args.video)

    print("walk slowly, keep at least one already-seen marker in frame "
          "while a new one comes into view. 'q' to stop and solve.")
    if not args.no_board:
        print("include the LANDING BOARD in at least one frame alongside a "
              "room marker, or the map cannot place it.")
    for frame in frames:
        if frame is None:
            continue
        n = mapper.add_frame(frame)
        if args.view:
            corners, ids, _ = mapper.detector.detectMarkers(
                frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            view = frame.copy()
            cv2.aruco.drawDetectedMarkers(view, corners, ids)
            cv2.putText(view, f"frames used {mapper.frames_used}  seen {n}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("map_room", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"\nframes used: {mapper.frames_used}")
    print("seen: " + str({_node_name(i): mapper.seen_count[i]
                          for i in sorted(mapper.seen_count)}))

    room_map, root_id = mapper.build(args.root_id)
    print(f"root marker: {root_id} (world origin)")
    print(f"solved {len(room_map)} marker(s): {sorted(room_map.markers)}")
    if room_map.has_board:
        print(f"landing board at {np.round(room_map.board_t * 100, 1)} cm "
              "in room frame")
    room_map.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
