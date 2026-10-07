"""
3D view of the room map and the drone inside it.

Pure rendering, exactly like tello_map.py: a RoomMap and a position in,
a BGR image out. No drone, no windows, no threads - so the GUI can put it
in a tab and anything else can save a still.

Where tello_map.py draws the two orthographic panels that matter during
an approach, this draws the whole room in perspective, which is the view
that tells you whether the MAP you just built is actually shaped like the
room you are standing in. A marker chained through a bad edge lands
somewhere obviously wrong here, while looking perfectly healthy as a
number.

    python room_view.py --map room_map.npz            # still, orbiting
    python room_view.py --map room_map.npz --save room.png

FRAME
    Room frame (W) from room_map.py: +Y up, the rest fixed by the root
    marker. Distances are metres internally, centimetres at the edges,
    matching RoomPose.p_world_cm.
"""

import argparse
import math

import cv2
import numpy as np

from room_map import RoomMap
from tello_map import (C_BAD, C_BG, C_BOARD, C_DIM, C_DRONE, C_GHOST, C_GRID,
                       C_PAD, C_SP, C_TEXT, C_TRAIL, _dashed, _text)
from tello_pose import BoardGeometry

C_MARKER = (200, 170, 90)      # room markers, cool blue-grey
C_FLOOR = (60, 60, 68)


class RoomView3D:
    """Orbiting perspective camera over the room map.

    Painter's algorithm, no z-buffer: every item is a flat quad or a line,
    so sorting by centroid depth is enough and costs nothing.
    """

    def __init__(self, geom=BoardGeometry()):
        self.geom = geom
        self.az = math.radians(35.0)
        self.el = math.radians(22.0)
        self.dist = 4.0
        self.target = np.zeros(3)
        self._framed_n = -1
        self._user_moved = False

    # -- camera ---------------------------------------------------------
    def orbit(self, d_az_deg, d_el_deg):
        self.az += math.radians(d_az_deg)
        self.el = float(np.clip(self.el + math.radians(d_el_deg),
                                math.radians(-80.0), math.radians(80.0)))
        self._user_moved = True

    def frame_points(self, pts_m, min_dist=1.4):
        """Aim at the middle of these world points (metres), far enough back
        to see them all - for a scene with no room map, e.g. just the board,
        the pad and the approach in front of it."""
        pts = np.asarray(pts_m, float)
        self.target = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        spread = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        self.dist = max(min_dist, spread * 1.05)
        self._framed_n = 0

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist * factor, 0.6, 30.0))
        self._user_moved = True

    def frame_map(self, room_map, force=False):
        """Point the camera at whatever the map contains.

        Re-frames whenever the map grows, because a map being built starts
        as one marker and framing on that leaves everything that connects
        later off the side of the view. Stops once the user has dragged:
        re-framing under someone's cursor is infuriating.
        """
        if room_map is None or (self._user_moved and not force):
            return
        n = len(room_map.markers) + (1 if room_map.has_board else 0)
        pts = [m.t for m in room_map.markers.values()]
        if room_map.has_board:
            pts.append(room_map.board_t)
            # The pad hangs pad_drop below the board and pad_out in front
            # of it; leaving it out framed the scene so tight that the pad
            # and the drone above it fell off the bottom of the view.
            pts.append(room_map.board_to_world_cm(
                [0.0, -self.geom.pad_drop_m * 100.0,
                 self.geom.pad_out_m * 100.0]) / 100.0)
        if not pts:
            return
        pts = np.array(pts, float)
        target = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        spread = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        dist = max(3.0, spread * 2.5)

        # Trigger on the geometry, not on the marker count. A map being
        # built can keep the same number of markers and still land in a
        # completely different frame, because build() re-picks the root
        # and the root IS the origin - watching the count alone left the
        # camera pointed at where the map used to be.
        if not force and self._framed_n >= 0:
            if (float(np.linalg.norm(target - self.target)) < 0.25 * self.dist
                    and abs(dist - self.dist) < 0.25 * self.dist):
                return
        self.target, self.dist = target, dist
        self._framed_n = n

    def _eye(self):
        d = np.array([math.cos(self.el) * math.sin(self.az),
                      math.sin(self.el),
                      math.cos(self.el) * math.cos(self.az)])
        return self.target + d * self.dist

    def _basis(self, eye):
        fwd = self.target - eye
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(right)
        if n < 1e-6:                       # looking straight down the up axis
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / n
        up = np.cross(right, fwd)
        return np.stack([right, -up, fwd])          # rows: x, y(down), z

    # -- projection ------------------------------------------------------
    def _project(self, pts, R, eye, f, w, h):
        """Nx3 world -> (Nx2 pixels, valid mask, depths)."""
        cam = (R @ (np.asarray(pts, float) - eye).T).T
        z = cam[:, 2]
        ok = z > 0.05
        zz = np.where(ok, z, 1.0)
        uv = np.empty((len(cam), 2))
        uv[:, 0] = f * cam[:, 0] / zz + w / 2.0
        uv[:, 1] = f * cam[:, 1] / zz + h / 2.0
        return uv, ok, z

    # -- pieces ----------------------------------------------------------
    def _floor_y(self, room_map):
        if room_map is not None and room_map.has_board:
            return float(room_map.board_t[1]) - self.geom.pad_drop_m
        ys = [float(m.t[1]) for m in (room_map.markers.values()
                                      if room_map else [])]
        return (min(ys) - 1.0) if ys else -1.0

    def _draw_grid(self, img, R, eye, f, w, h, y, step=0.5, half=None):
        # Size the floor to the room, not to a fixed guess: a 2 m map under
        # an 8 m grid reads as a speck in an empty field.
        if half is None:
            half = float(np.clip(self.dist * 0.75, 1.5, 8.0))
        segs = []
        cx, cz = self.target[0], self.target[2]
        x0 = math.floor((cx - half) / step) * step
        z0 = math.floor((cz - half) / step) * step
        for i in range(int(2 * half / step) + 1):
            x = x0 + i * step
            segs.append(([x, y, z0], [x, y, z0 + 2 * half]))
            z = z0 + i * step
            segs.append(([x0, y, z], [x0 + 2 * half, y, z]))
        for a, b in segs:
            pts, ok, _ = self._project([a, b], R, eye, f, w, h)
            if ok.all():
                cv2.line(img, tuple(np.int32(pts[0])), tuple(np.int32(pts[1])),
                         C_GRID, 1, cv2.LINE_AA)

    def _quad(self, img, corners, R, eye, f, w, h, colour, label=None):
        """A marker's true quad plus a minimum-size dot. At room scale a
        15 cm marker is a few pixels across and the quad alone vanishes,
        but the quad is what shows which way the marker faces - so draw
        both and hang the label above, clear of the geometry."""
        pts, ok, _ = self._project(corners, R, eye, f, w, h)
        if not ok.all():
            return
        poly = np.int32(pts).reshape(-1, 1, 2)
        shade = tuple(int(c * 0.35) for c in colour)
        cv2.fillConvexPoly(img, poly, shade, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, colour, 2, cv2.LINE_AA)
        c = pts.mean(axis=0)
        span = float(np.abs(pts - c).max())
        if span < 4.0:
            cv2.circle(img, tuple(np.int32(c)), 4, colour, 1, cv2.LINE_AA)
        if label:
            _text(img, label,
                  (int(c[0]) - 4 * len(label), int(c[1] - max(span, 4) - 6)),
                  colour, 0.42)

    def _board_corners(self, room_map):
        s = self.geom.span_m / 2.0
        local = np.array([[-s, s, 0.0], [s, s, 0.0],
                          [s, -s, 0.0], [-s, -s, 0.0]])
        return (room_map.board_R @ local.T).T + room_map.board_t

    def _pad_ring(self, room_map, n=24, r=0.25):
        c = room_map.board_to_world_cm([0.0,
                                        -self.geom.pad_drop_m * 100.0,
                                        self.geom.pad_out_m * 100.0]) / 100.0
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([c[0] + r * np.cos(a),
                         np.full(n, c[1]),
                         c[2] + r * np.sin(a)], axis=1), c

    # -- public ----------------------------------------------------------
    def render(self, w, h, room_map, p_world_cm=None, R_world_cam=None,
               trail=(), seen=True, hud=(), setpoint_cm=None):
        img = np.full((h, w, 3), C_BG, np.uint8)
        f = 0.9 * w
        eye = self._eye()
        R = self._basis(eye)

        hud_h = 16 * len(hud) + (8 if hud else 0)
        floor = self._floor_y(room_map)
        self._draw_grid(img, R, eye, f, w, h - hud_h, floor)

        items = []          # (depth, drawer)
        if room_map is not None:
            for mid, m in room_map.markers.items():
                c = m.object_points_world()
                items.append((np.linalg.norm(m.t - eye),
                              lambda i, c=c, mid=mid: self._quad(
                                  i, c, R, eye, f, w, h - hud_h,
                                  C_MARKER, str(mid))))
            if room_map.has_board:
                c = self._board_corners(room_map)
                items.append((np.linalg.norm(room_map.board_t - eye),
                              lambda i, c=c: self._quad(
                                  i, c, R, eye, f, w, h - hud_h,
                                  C_BOARD, "board")))
                ring, pc = self._pad_ring(room_map)
                items.append((np.linalg.norm(pc - eye),
                              lambda i, ring=ring: self._ring(
                                  i, ring, R, eye, f, w, h - hud_h, C_PAD)))

        if room_map is not None and room_map.has_board:
            # the board's stand, so its height above the floor reads in 3D
            bottom = room_map.board_to_world_cm(
                [0.0, -self.geom.span_m * 50.0, 0.0]) / 100.0
            foot = np.array([bottom[0], floor, bottom[2]])
            pts, ok, _ = self._project([bottom, foot], R, eye, f, w, h - hud_h)
            if ok.all():
                cv2.line(img, tuple(np.int32(pts[0])), tuple(np.int32(pts[1])),
                         tuple(int(v * 0.55) for v in C_BOARD), 2, cv2.LINE_AA)

        for _, draw in sorted(items, key=lambda it: -it[0]):
            draw(img)

        if len(trail) > 1:
            t = np.asarray(trail, float) / 100.0
            pts, ok, _ = self._project(t, R, eye, f, w, h - hud_h)
            good = np.int32(pts[ok])
            if len(good) > 1:
                cv2.polylines(img, [good.reshape(-1, 1, 2)], False,
                              C_TRAIL, 1, cv2.LINE_AA)

        if setpoint_cm is not None:
            self._draw_setpoint(img, np.asarray(setpoint_cm, float) / 100.0,
                                None if p_world_cm is None
                                else np.asarray(p_world_cm, float) / 100.0,
                                R, eye, f, w, h - hud_h, floor)

        if p_world_cm is not None:
            self._draw_drone(img, np.asarray(p_world_cm, float) / 100.0,
                             R_world_cam, R, eye, f, w, h - hud_h, floor, seen)
        else:
            _text(img, "no fix", (w // 2 - 24, (h - hud_h) // 2), C_BAD, 0.6, 2)

        for i, line in enumerate(hud):
            _text(img, line, (10, h - hud_h + 14 + 16 * i), C_TEXT, 0.42)
        return img

    def _ring(self, img, ring, R, eye, f, w, h, colour):
        pts, ok, _ = self._project(ring, R, eye, f, w, h)
        if ok.all():
            cv2.polylines(img, [np.int32(pts).reshape(-1, 1, 2)], True,
                          colour, 1, cv2.LINE_AA)

    def _draw_setpoint(self, img, sp, drone, R, eye, f, w, h, floor):
        """Magenta ring where AUTO HOLD is headed, a drop line to the floor
        so its height reads, and a dashed line from the drone to it."""
        pts, ok, _ = self._project([sp, [sp[0], floor, sp[2]]], R, eye, f, w, h)
        if not ok[0]:
            return
        c = tuple(np.int32(pts[0]))
        if ok[1]:
            b = tuple(np.int32(pts[1]))
            _dashed(img, c, b, tuple(int(v * 0.6) for v in C_SP), 1, 3, 4)
            cv2.circle(img, b, 3, tuple(int(v * 0.6) for v in C_SP), 1, cv2.LINE_AA)
        if drone is not None:
            d, okd, _ = self._project([drone], R, eye, f, w, h)
            if okd[0]:
                _dashed(img, tuple(np.int32(d[0])), c, C_SP, 1, 3, 5)
        cv2.circle(img, c, 10, C_SP, 1, cv2.LINE_AA)
        cv2.drawMarker(img, c, C_SP, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        _text(img, "SP", (c[0] + 12, c[1] - 8), C_SP, 0.38)

    def _draw_drone(self, img, p, R_world_cam, R, eye, f, w, h, floor, seen):
        col = C_DRONE if seen else C_GHOST
        pts, ok, _ = self._project([p, [p[0], floor, p[2]]], R, eye, f, w, h)
        if not ok[0]:
            return
        c = tuple(np.int32(pts[0]))
        if ok[1]:       # drop line to the floor, so height is readable
            b = tuple(np.int32(pts[1]))
            cv2.line(img, c, b, tuple(int(v * 0.5) for v in col), 1, cv2.LINE_AA)
            cv2.circle(img, b, 2, tuple(int(v * 0.5) for v in col), -1, cv2.LINE_AA)
        cv2.circle(img, c, 6, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, 6, (20, 20, 24), 1, cv2.LINE_AA)
        if R_world_cam is not None:
            fwd = np.asarray(R_world_cam, float)[:, 2]
            tip, ok2, _ = self._project([p + fwd * 0.45], R, eye, f, w, h)
            if ok2[0]:
                cv2.arrowedLine(img, c, tuple(np.int32(tip[0])), col, 2,
                                cv2.LINE_AA, tipLength=0.3)


def main():
    ap = argparse.ArgumentParser(description="still 3D view of a room map")
    ap.add_argument("--map", default="room_map.npz")
    ap.add_argument("--save", metavar="PNG")
    ap.add_argument("--size", default="700x560")
    args = ap.parse_args()

    room_map = RoomMap.load(args.map)
    w, h = (int(v) for v in args.size.lower().split("x"))
    from board_config import load_board
    view = RoomView3D(load_board())
    view.frame_map(room_map)
    hud = [str(room_map)]

    if args.save:
        cv2.imwrite(args.save, view.render(w, h, room_map, hud=hud))
        print(f"wrote {args.save}")
        return

    print("drag is GUI-only here; this window just spins. q to quit.")
    while True:
        view.orbit(1.0, 0.0)
        cv2.imshow("room", view.render(w, h, room_map, hud=hud))
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
