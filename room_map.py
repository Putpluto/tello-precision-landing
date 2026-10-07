"""
Multi-marker room map: world-frame pose of every ArUco marker placed
around the room, built automatically by map_room.py from a walkthrough
video (no manual measurement needed - see map_room.py's docstring).

World frame (W): fixed to whichever marker map_room.py picked as the
root at build time. +Z out of that marker's printed face, +Y up, +X
right when facing it - same convention as BoardGeometry's board frame,
because the root marker's local frame *is* the world frame.

MOUNT THE ROOM MARKERS UPRIGHT. World +Y is taken to be vertical by
everything that draws or flies on this map. A root marker mounted on its
side tilts the whole room.

ID SPACE: ids 0-3 are reserved for the landing board (BoardGeometry in
tello_pose.py). Room markers must use ids 4+ from the same DICT_4X4_50
dictionary, so a single frame that happens to see both never collides.

THE LANDING BOARD is stored separately from the markers, as one rigid
body (board_R/board_t) rather than four independent markers, because its
internal geometry is already pinned down by BoardGeometry. That pose is
the bridge between this map and the board frame the landing controller
works in (board_to_world_cm / world_to_board_cm). The mission does not fly
on the room map yet; the GUI draws and localizes with it.
"""

from dataclasses import dataclass

import numpy as np

ROOM_ID_MIN = 4          # 0-3 reserved for the landing board
BOARD_NODE = -1          # graph-node key for the landing board as a whole


@dataclass
class MarkerPose:
    id: int
    size_m: float
    R: np.ndarray         # 3x3, this marker's axes expressed in world frame
    t: np.ndarray         # 3,   this marker's origin in world frame

    def object_points_world(self):
        """4x3 corners in world frame, ordered TL,TR,BR,BL - matches
        BoardGeometry.object_points() so the same PnP code works on
        either."""
        h = self.size_m / 2.0
        local = np.array([
            [-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0],
        ], dtype=np.float64)
        return (self.R @ local.T).T + self.t


class RoomMap:
    def __init__(self, markers=None, board_R=None, board_t=None):
        self.markers = dict(markers) if markers else {}   # id -> MarkerPose
        self.board_R = None if board_R is None else np.asarray(board_R, float)
        self.board_t = None if board_t is None else np.asarray(board_t, float)

    def object_points_world(self, ids):
        return {i: self.markers[i].object_points_world()
                for i in ids if i in self.markers}

    # -- landing board bridge ------------------------------------------
    @property
    def has_board(self):
        return self.board_R is not None and self.board_t is not None

    def board_to_world_cm(self, p_board_cm):
        """A point in board frame (cm, as the landing controller thinks)
        expressed in room frame (cm). None if the board was never mapped."""
        if not self.has_board:
            return None
        p_m = np.asarray(p_board_cm, float) / 100.0
        return (self.board_R @ p_m + self.board_t) * 100.0

    def world_to_board_cm(self, p_world_cm):
        if not self.has_board:
            return None
        p_m = np.asarray(p_world_cm, float) / 100.0
        return (self.board_R.T @ (p_m - self.board_t)) * 100.0

    # -- io ------------------------------------------------------------
    def save(self, path):
        ids = sorted(self.markers)
        if not ids:
            raise ValueError("empty room map - nothing to save")
        kw = dict(
            ids=np.array(ids, dtype=np.int32),
            sizes=np.array([self.markers[i].size_m for i in ids]),
            R=np.stack([self.markers[i].R for i in ids]),
            t=np.stack([self.markers[i].t for i in ids]),
        )
        if self.has_board:
            kw["board_R"] = self.board_R
            kw["board_t"] = self.board_t
        np.savez(path, **kw)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        markers = {}
        for k, mid in enumerate(d["ids"]):
            mid = int(mid)
            markers[mid] = MarkerPose(
                id=mid, size_m=float(d["sizes"][k]),
                R=d["R"][k].astype(np.float64),
                t=d["t"][k].astype(np.float64),
            )
        board_R = d["board_R"].astype(np.float64) if "board_R" in d else None
        board_t = d["board_t"].astype(np.float64) if "board_t" in d else None
        return cls(markers, board_R, board_t)

    def __len__(self):
        return len(self.markers)

    def __repr__(self):
        ids = sorted(self.markers)
        board = "with board" if self.has_board else "no board"
        return f"RoomMap({len(ids)} markers: {ids}, {board})"
