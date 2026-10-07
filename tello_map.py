"""
Localization map for the ArUco landing board.

Pure rendering: positions in, BGR image out. No drone, no windows, no
threads, so one renderer serves the live GUI, an offline replay of a
--log CSV, and a still PNG for a report.

    python tello_map.py --demo                     # synthetic, animated
    python tello_map.py --log run.csv              # replay a real flight
    python tello_map.py --log run.csv --video run.mp4   # both, side by side
    python tello_map.py --video run.mp4            # estimate from the video
    python tello_map.py --demo --save map.png      # one frame, headless
    python tello_map.py --log run.csv --video run.mp4 --save replay.mp4

--log and --record from the same run pair up: the controller writes one CSV
row and one video frame per loop, so row i is frame i. Play them together
and you can see what the camera saw at the moment the map went wrong.
SPACE pauses, any other key steps one frame, q quits.

FRAME (board frame M, identical to tello_pose.py)
    Origin at the centre of the cross-shaped gap between the four markers,
    +X right as you face the print, +Y up, +Z out of the face into the
    room. Drone position p = (px, py, pz), centimetres.

PANELS
    TOP-DOWN (x-z): board along the top edge, range grows downward. This is
        the view that matters during the approach - lateral error left and
        right, range up and down, heading as the nose of the drone.
    SIDE (z-y): board face on the left, pad surface as the floor line. This
        is the view that matters for height and for the final descent.

The map is drawn from the SAME numbers the controller acts on, so if the
map looks wrong the pose is wrong - it is a debugging instrument, not
decoration.
"""

import argparse
import csv
import math

import cv2
import numpy as np

from tello_io import put_text
from tello_pose import BoardGeometry

# BGR palette. Deliberately dim background so the coloured geometry reads
# clearly next to a bright video panel.
C_BG     = (26, 26, 30)
C_GRID   = (52, 52, 60)
C_AXIS   = (88, 88, 98)
C_TEXT   = (220, 220, 226)
C_DIM    = (142, 142, 152)
C_BOARD  = (60, 190, 255)      # amber
C_PAD    = (110, 210, 130)     # green
C_DRONE  = (255, 196, 74)      # cyan
C_GHOST  = (110, 110, 120)     # last known position, no current fix
C_TRAIL  = (180, 140, 70)
C_STAND  = (150, 130, 240)     # standoff rings
C_FOV    = (96, 86, 58)
C_BAD    = (70, 70, 235)
C_SP     = (255, 90, 255)      # setpoint, magenta
C_LIM    = (110, 64, 110)      # where a setpoint is allowed (dim magenta)


def _text(img, s, org, color=C_TEXT, scale=0.40, thick=1):
    put_text(img, s, org, scale, color, thick, cv2.LINE_AA)


def _dashed(img, p0, p1, color, thick=1, dash=8, gap=6):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.hypot(*d))
    if length < 1.0:
        return
    u = d / length
    t = 0.0
    while t < length:
        a = p0 + u * t
        b = p0 + u * min(t + dash, length)
        cv2.line(img, tuple(np.round(a).astype(int)),
                 tuple(np.round(b).astype(int)), color, thick, cv2.LINE_AA)
        t += dash + gap


class LocalizationMap:
    """Two orthographic panels of the drone in board frame.

    Everything is in centimetres to match BoardPose.p_board_cm, so no unit
    conversion happens between the controller and what you see.
    """

    def __init__(self, geom=BoardGeometry(), z_max=250.0, y_half=45.0,
                 standoffs=(("FAR", 150.0), ("NEAR", 70.0)),
                 fov_deg=55.0, pad_radius_cm=25.0):
        # y_half is the minimum headroom shown above board centre in the
        # side view; the drone flies near y = 0 (H_SET), so a little is
        # plenty and asking for more only shrinks the range axis.
        self.geom = geom
        self.z_max = float(z_max)
        self.y_half = float(y_half)
        self.standoffs = tuple(standoffs)
        self.fov = float(fov_deg)
        self.pad_r = float(pad_radius_cm)
        self.pad_drop = geom.pad_drop_m * 100.0
        self.pad_out = geom.pad_out_m * 100.0
        self.span = geom.span_m * 100.0      # 16.5 cm board, or the one marker
        self.x_half = 90.0        # cm guaranteed visible either side
        self._last = None         # (w, h, n_hud) of the last render, for to_world

    # -- panel geometry: one place, shared by drawing and clicking --------
    @staticmethod
    def _split(h, n_hud):
        """Rows of the top-down panel, and where the side panel ends."""
        hud_h = 16 * n_hud + (8 if n_hud else 0)
        return int((h - hud_h) * 0.60), h - hud_h

    def _top_xf(self, w, h):
        """TOP-DOWN panel w x h: (x, z) cm -> pixel (ox + x*s, oy + z*s)."""
        m_top, m_bot, m_side = 30, 18, 30
        s = (h - m_top - m_bot) / self.z_max
        s = min(s, (w / 2.0 - m_side) / self.x_half)
        return w / 2.0, float(m_top), s

    def _side_xf(self, w, h):
        """SIDE panel w x h: (z, y) cm -> pixel (ox + z*s, oy - y*s).
        Isotropic, but range fills the width and y gets whatever height
        that leaves. Scaling off the y extent instead pads the range axis
        out past 4 m and squeezes pad and standoffs into the left third."""
        m_l, m_r, m_t, m_b = 36, 14, 26, 26
        y_lo = -(self.pad_drop + 30.0)
        s = (w - m_l - m_r) / self.z_max
        y_span = (h - m_t - m_b) / s
        min_span = -y_lo + self.y_half
        if y_span < min_span:
            s = (h - m_t - m_b) / min_span
            y_span = min_span
        return float(m_l), m_t + (y_lo + y_span) * s, s, y_lo, y_span

    def to_world(self, u, v, size=None):
        """A pixel of a render()ed image -> ("top", x, z) or ("side", z, y)
        in cm, or None outside both panels. size = (w, h, n_hud_lines);
        default: the last render, which is what a click landed on."""
        if size is None:
            if self._last is None:
                return None
            size = self._last
        w, h, n_hud = size
        h_top, h_end = self._split(h, n_hud)
        if 0 <= v < h_top:
            ox, oy, s = self._top_xf(w, h_top)
            return "top", (u - ox) / s, (v - oy) / s
        if h_top <= v < h_end:
            ox, oy, s, _, _ = self._side_xf(w, h_end - h_top)
            return "side", (u - ox) / s, (oy - (v - h_top)) / s
        return None

    def to_pixel(self, panel, a, b, size):
        """Inverse of to_world: ("top", x, z) or ("side", z, y) -> (u, v)."""
        w, h, n_hud = size
        h_top, h_end = self._split(h, n_hud)
        if panel == "top":
            ox, oy, s = self._top_xf(w, h_top)
            return ox + a * s, oy + b * s
        ox, oy, s, _, _ = self._side_xf(w, h_end - h_top)
        return ox + a * s, h_top + oy - b * s

    # -- top-down ------------------------------------------------------
    def _top(self, img, p, yaw, trail, seen, setpoint=None, limits=None):
        h, w = img.shape[:2]
        m_top, m_bot = 30, 18
        ox, oy, s = self._top_xf(w, h)

        def P(x, z):
            return (int(round(ox + x * s)), int(round(oy + z * s)))

        x_vis = (w / 2.0 - 4) / s
        z_vis = (h - m_top - m_bot) / s

        step = 50
        for i in range(-int(x_vis // step), int(x_vis // step) + 1):
            u = P(i * step, 0)[0]
            cv2.line(img, (u, m_top), (u, h - m_bot),
                     C_AXIS if i == 0 else C_GRID, 1)
            if i:
                _text(img, f"{i*step:+d}", (u - 11, h - 6), C_DIM, 0.33)
        for j in range(0, int(z_vis // step) + 1):
            v = P(0, j * step)[1]
            cv2.line(img, (2, v), (w - 2, v), C_AXIS if j == 0 else C_GRID, 1)
            if j:
                _text(img, f"{j*step}", (4, v - 4), C_DIM, 0.33)

        for name, z in self.standoffs:
            if z > z_vis:
                continue
            v = P(0, z)[1]
            _dashed(img, (2, v), (w - 2, v), C_STAND, 1)
            _text(img, f"{name} {z:.0f}", (w - 72, v - 5), C_STAND, 0.34)

        # pad, seen from above: a disc centred pad_out out from the face
        pc = P(0.0, self.pad_out)
        cv2.circle(img, pc, max(3, int(round(self.pad_r * s))), C_PAD, 1,
                   cv2.LINE_AA)
        cv2.drawMarker(img, pc, C_PAD, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        _text(img, "PAD", (pc[0] + int(self.pad_r * s) + 5, pc[1] + 4),
              C_PAD, 0.36)

        # board: edge-on from above, plus its face normal
        half = self.span / 2.0
        cv2.line(img, P(-half, 0), P(half, 0), C_BOARD, 4, cv2.LINE_AA)
        _dashed(img, P(0, 0), P(0, 20), C_BOARD, 1, 5, 5)
        _text(img, "BOARD", (P(half, 0)[0] + 6, m_top + 5), C_BOARD, 0.36)
        _text(img, "TOP-DOWN   x / z", (10, 17), C_DIM, 0.40)

        if len(trail) > 1:
            pts = [P(t[0], t[2]) for t in trail]
            n = len(pts)
            for i in range(1, n):
                f = i / n                       # fade the tail out
                col = tuple(int(c * (0.25 + 0.75 * f)) for c in C_TRAIL)
                cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

        if limits is not None:                  # where a setpoint may go
            pts = [P(x, z) for x, z in self.allowed_top(limits)]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                _dashed(img, a, b, C_LIM, 1, 3, 5)
        if setpoint is not None:
            self._draw_setpoint(img, P(setpoint[0], setpoint[2]),
                                None if p is None else P(p[0], p[2]))

        if p is None:
            _text(img, "NO FIX", (w // 2 - 30, h // 2), C_BAD, 0.60, 2)
            return

        c = np.array(P(p[0], p[2]), float)
        yr = math.radians(yaw)
        # yaw 0 = square on, so forward is -Z_M, which is up the panel
        fwd = np.array([math.sin(yr), -math.cos(yr)])
        perp = np.array([-fwd[1], fwd[0]])
        col = C_DRONE if seen else C_GHOST

        # camera FOV wedge - shows whether the board is actually in view
        reach = max(40.0, abs(p[2]) + 25.0) * s
        for sign in (-1.0, 1.0):
            a = yr + sign * math.radians(self.fov / 2.0)
            d = np.array([math.sin(a), -math.cos(a)])
            _dashed(img, c, c + d * reach, C_FOV, 1, 5, 7)

        r = 9.0
        tri = np.array([c + fwd * r,
                        c - fwd * r * 0.7 + perp * r * 0.62,
                        c - fwd * r * 0.7 - perp * r * 0.62])
        cv2.fillConvexPoly(img, tri.astype(np.int32), col, cv2.LINE_AA)
        cv2.polylines(img, [tri.astype(np.int32)], True, (20, 20, 24), 1,
                      cv2.LINE_AA)

    # -- setpoints -------------------------------------------------------
    @staticmethod
    def _draw_setpoint(img, c, drone=None):
        """A ring wider than the drone icon, so it stays visible once the
        drone is sitting on it, and a dashed line from where it is now."""
        if drone is not None:
            _dashed(img, drone, c, C_SP, 1, 3, 5)
        cv2.circle(img, c, 11, C_SP, 1, cv2.LINE_AA)
        cv2.drawMarker(img, c, C_SP, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        _text(img, "SP", (c[0] + 13, c[1] - 9), C_SP, 0.38)

    @staticmethod
    def allowed_top(lim):
        """Outline, (x, z) cm, of where a setpoint may go seen from above
        (landing_control.SetpointLimits): in range, not too far off-axis."""
        t = math.tan(math.radians(lim.max_view_deg))
        return [(-t * lim.z_min, lim.z_min), (t * lim.z_min, lim.z_min),
                (t * lim.z_max, lim.z_max), (-t * lim.z_max, lim.z_max)]

    def allowed_side(self, lim, n=24):
        """Outline, (z, y) cm, seen from the side (straight in front of the
        board): the board inside the camera's vertical view, clear of the
        floor. Off-axis it widens a little - clamp_setpoint is exact."""
        t = math.tan(math.radians(lim.max_elev_deg))
        floor = -self.pad_drop + lim.floor_clear
        zs = np.linspace(lim.z_min, lim.z_max, n)
        top = [(z, min(t * z, lim.y_max)) for z in zs]
        bot = [(z, max(-t * z, floor)) for z in zs[::-1]]
        return top + bot

    # -- side ----------------------------------------------------------
    def _side(self, img, p, trail, seen, setpoint=None, limits=None):
        h, w = img.shape[:2]
        m_l, m_r, m_t, m_b = 36, 14, 26, 26
        ox, oy, s, y_lo, y_span = self._side_xf(w, h)

        def P(z, y):
            return (int(round(ox + z * s)), int(round(oy - y * s)))

        z_vis = (w - m_l - m_r) / s
        step = 50
        for j in range(0, int(z_vis // step) + 1):
            u = P(j * step, 0)[0]
            cv2.line(img, (u, m_t), (u, h - m_b),
                     C_AXIS if j == 0 else C_GRID, 1)
            if j:
                _text(img, f"{j*step}", (u - 10, h - 8), C_DIM, 0.33)
        for yv in (-100, -50, 0, 50, 100):
            if not (y_lo <= yv <= y_lo + y_span):
                continue
            v = P(0, yv)[1]
            cv2.line(img, (2, v), (w - 2, v), C_AXIS if yv == 0 else C_GRID, 1)
            _text(img, f"{yv:+d}", (2, v - 4), C_DIM, 0.33)

        for _, z in self.standoffs:
            if z <= z_vis:
                _dashed(img, (P(z, 0)[0], m_t), (P(z, 0)[0], h - m_b),
                        C_STAND, 1)

        # pad surface: the floor the drone actually lands on
        floor = P(0, -self.pad_drop)[1]
        cv2.line(img, (P(0, 0)[0], floor),
                 (P(min(self.pad_out + 45, z_vis), 0)[0], floor),
                 (70, 120, 85), 1, cv2.LINE_AA)
        cv2.line(img, P(max(self.pad_out - self.pad_r, 0), -self.pad_drop),
                 P(self.pad_out + self.pad_r, -self.pad_drop), C_PAD, 3,
                 cv2.LINE_AA)
        cv2.drawMarker(img, P(self.pad_out, -self.pad_drop), C_PAD,
                       cv2.MARKER_TRIANGLE_DOWN, 9, 1, cv2.LINE_AA)

        # board face, seen edge-on from the side
        half = self.span / 2.0
        cv2.line(img, P(0, -half), P(0, half), C_BOARD, 4, cv2.LINE_AA)
        _text(img, "SIDE   z / y", (10, 15), C_DIM, 0.40)

        if len(trail) > 1:
            pts = [P(t[2], t[1]) for t in trail]
            n = len(pts)
            for i in range(1, n):
                f = i / n
                col = tuple(int(c * (0.25 + 0.75 * f)) for c in C_TRAIL)
                cv2.line(img, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

        if limits is not None:
            pts = [P(z, y) for z, y in self.allowed_side(limits)]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                _dashed(img, a, b, C_LIM, 1, 3, 5)
        if setpoint is not None:
            self._draw_setpoint(img, P(setpoint[2], setpoint[1]),
                                None if p is None else P(p[2], p[1]))

        if p is None:
            return
        c = P(p[2], p[1])
        col = C_DRONE if seen else C_GHOST
        _dashed(img, c, (c[0], floor), col, 1, 4, 5)   # height above the pad
        cv2.circle(img, c, 6, col, -1, cv2.LINE_AA)
        cv2.circle(img, c, 6, (20, 20, 24), 1, cv2.LINE_AA)

    # -- public --------------------------------------------------------
    def render(self, w=460, h=560, p=None, yaw=0.0, trail=(), seen=True,
               hud=(), setpoint=None, limits=None):
        """p: (px, py, pz) cm or None. trail: iterable of (px, py, pz).
        setpoint: (x, y, z) cm to mark in both panels. limits: a
        landing_control.SetpointLimits, to outline where one may go."""
        img = np.full((h, w, 3), C_BG, np.uint8)
        h_top, h_end = self._split(h, len(hud))
        self._last = (w, h, len(hud))
        self._top(img[:h_top], p, yaw, trail, seen, setpoint, limits)
        self._side(img[h_top:h_end], p, trail, seen, setpoint, limits)
        cv2.line(img, (0, h_top), (w, h_top), C_AXIS, 1)
        for i, line in enumerate(hud):
            _text(img, line, (10, h_end + 14 + 16 * i),
                  C_SP if line.startswith("setpoint") else C_TEXT, 0.42)
        return img

    def hud_lines(self, p, yaw=None, extra=None, setpoint=None):
        """Standard numeric readout to sit under the panels. Always the
        same number of lines for the same arguments, pose or no pose - the
        panels above are sized from it, and clicks are mapped through them."""
        if p is None:
            out = ["no pose", ""]
        else:
            px, py, pz = p
            out = [f"x {px:+7.1f}  y {py:+7.1f}  z {pz:7.1f} cm"
                   + (f"   yaw {yaw:+6.1f}" if yaw is not None else ""),
                   f"pad  dz {pz - self.pad_out:+7.1f}  "
                   f"height {py + self.pad_drop:6.1f} cm"]
        if setpoint is not None:
            sx, sy, sz = setpoint
            s = f"setpoint x {sx:+.0f}  y {sy:+.0f}  z {sz:.0f} cm"
            if p is not None:
                d = np.asarray(setpoint, float) - np.asarray(p, float)
                s += f"   to go {d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f}"
            out.append(s)
        if extra:
            out.append(extra)
        return out


# ----------------------------------------------------------------------
# Offline sources
# ----------------------------------------------------------------------

def demo_track(t):
    """Synthetic approach: far hold -> near hold -> hop. Returns (p, yaw)."""
    legs = [(0.0, (70.0, 35.0, 200.0)), (6.0, (0.0, 0.0, 150.0)),
            (9.0, (0.0, 0.0, 150.0)), (15.0, (0.0, 0.0, 70.0)),
            (18.0, (0.0, 0.0, 70.0)), (21.0, (0.0, -30.0, 28.0))]
    t = t % legs[-1][0]
    p = np.array(legs[-1][1], float)
    for i in range(len(legs) - 1):
        t0, a = legs[i]
        t1, b = legs[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / max(t1 - t0, 1e-6)
            f = f * f * (3 - 2 * f)                 # smoothstep
            p = np.array(a) + (np.array(b) - np.array(a)) * f
            break
    p = p + np.array([2.0 * math.sin(t * 2.1), 1.5 * math.sin(t * 1.7), 0.0])
    yaw = math.degrees(math.atan2(p[0], p[2])) * 0.7
    return p, yaw


def read_log(path):
    """Rows written by tello_aruco_landing.py --log.

    A row where the pose was lost comes back with p = None instead of being
    dropped, so row i still lines up with frame i of the matching --record
    video, and the dropouts show in the replay rather than being silently
    stitched over.
    """
    out = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            try:
                p = (float(row["px"]), float(row["py"]), float(row["pz"]))
                if any(math.isnan(v) for v in p):
                    p = None
            except (ValueError, KeyError, TypeError):
                p = None
            try:
                yaw = float(row["yaw"]) if row.get("yaw") else 0.0
            except ValueError:
                yaw = 0.0
            try:
                t = float(row["t"])
            except (ValueError, KeyError, TypeError):
                t = float(i)
            out.append((t, p, yaw, row.get("state", "")))
    return out


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, n, (fps if fps and fps > 1e-3 else 15.0)


def compose(view, mp):
    """Video left, map right, heights matched. view may be None."""
    if view is None:
        return mp
    h = mp.shape[0]
    scale = h / view.shape[0]
    vw = cv2.resize(view, (max(1, int(round(view.shape[1] * scale))), h))
    return np.hstack([vw, mp])


def steps_log_video(rows, cap, n_frames, offset=0):
    """Recorded video beside the CSV that was written alongside it.

    The controller writes one row and one frame per loop, so normally row i
    is frame i; if the counts disagree (an aborted run, a dropped frame)
    fall back to spreading the rows evenly over the frames rather than
    drifting further out of sync with every step.
    """
    same = (n_frames == len(rows))
    if not same:
        print(f"note: {len(rows)} rows vs {n_frames} frames - "
              "scaling proportionally; use --offset to nudge")
    cur = -1
    frame = None            # held across rows: with more rows than frames
                            # several rows share one frame, and re-reading
                            # would run the video off the end early
    for i, (t, p, yaw, st) in enumerate(rows):
        idx = i if same else int(round(i * (n_frames - 1) /
                                       max(len(rows) - 1, 1)))
        idx = max(0, min(n_frames - 1, idx + offset))
        if idx < cur or idx > cur + 8:          # seek only on a real jump
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cur = idx - 1
        while cur < idx:
            ok, f = cap.read()
            if not ok:
                break
            frame, cur = f, cur + 1
        yield frame, p, yaw, p is not None, f"{st}  t {t:.1f}s  #{i}"


def steps_video_only(cap, est):
    """No CSV: run the estimator over the frames and map what it returns."""
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corners, ids, _ = est.detect(frame)
        pose = est.estimate(frame, detection=(corners, ids))
        view = est.draw(frame, pose, corners, ids)
        p = None if pose is None else pose.p_board_cm
        yaw = 0.0 if pose is None else pose.yaw_deg
        tag = ("no board" if pose is None
               else f"rms {pose.reproj_rms_px:.2f} px  n {pose.n_markers}")
        yield view, p, yaw, pose is not None, f"{tag}  #{i}"
        i += 1


def steps_log_only(rows):
    for i, (t, p, yaw, st) in enumerate(rows):
        yield None, p, yaw, p is not None, f"{st}  t {t:.1f}s  #{i}"


def steps_demo(seconds=21.0, dt=0.04):
    for t in np.arange(0.0, seconds, dt):
        p, yaw = demo_track(float(t))
        yield None, p, yaw, True, f"demo  t {t:5.1f}s"


def play(steps, lm, w, h, fps=20.0, save=None, trail_len=200,
         title="localization map"):
    """Drive the renderer from a step generator.

    With --save it runs headless: a .mp4/.avi name records the whole
    replay, anything else writes the final frame as a still.
    """
    trail = []
    video_out = bool(save) and save.lower().endswith((".mp4", ".avi"))
    still_out = bool(save) and not video_out
    writer = None
    last = None
    paused = False
    delay = max(1, int(round(1000.0 / max(fps, 1e-3))))
    n = 0

    for frame, p, yaw, seen, label in steps:
        if p is not None:
            trail.append(tuple(float(v) for v in p))
        n += 1
        last = (frame, p, yaw, seen, label)
        if still_out:
            continue                      # only the final frame is drawn
        mp = lm.render(w, h, p, yaw, trail[-trail_len:], seen,
                       lm.hud_lines(p, yaw, label))
        img = compose(frame, mp)
        if video_out:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(
                    *("mp4v" if save.lower().endswith(".mp4") else "MJPG"))
                writer = cv2.VideoWriter(save, fourcc, fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
            continue
        cv2.imshow(title, img)
        k = cv2.waitKey(0 if paused else delay) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused

    if still_out and last is not None:
        frame, p, yaw, seen, label = last
        mp = lm.render(w, h, p, yaw, trail[-trail_len:], seen,
                       lm.hud_lines(p, yaw, f"{label}  ({n} steps)"))
        cv2.imwrite(save, compose(frame, mp))
        print(f"wrote {save}")
    elif writer is not None:
        writer.release()
        print(f"wrote {save}  ({n} frames)")
    else:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="localization map viewer")
    ap.add_argument("--log", help="CSV from tello_aruco_landing.py --log")
    ap.add_argument("--video", help="mp4 from tello_aruco_landing.py --record")
    ap.add_argument("--demo", action="store_true", help="synthetic approach")
    ap.add_argument("--save", metavar="FILE",
                    help="headless: .mp4/.avi records the replay, "
                         "any other name writes the final frame as a still")
    ap.add_argument("--size", default="520x640",
                    help="map panel WxH, default 520x640")
    ap.add_argument("--fps", type=float,
                    help="playback rate, default the video's or 20")
    ap.add_argument("--offset", type=int, default=0,
                    help="shift the video by N frames to fix sync")
    ap.add_argument("--calib", default="tello_calib.npz",
                    help="intrinsics, only used for --video without --log")
    ap.add_argument("--trail", type=int, default=200)
    args = ap.parse_args()

    if not (args.log or args.video or args.demo):
        raise SystemExit("pick --demo, --log FILE.csv, and/or --video FILE.mp4")

    w, h = (int(v) for v in args.size.lower().split("x"))
    from board_config import load_board
    from landing_control import for_board
    geom = load_board()
    cfg, _ = for_board(geom)
    lm = LocalizationMap(geom, standoffs=(("FAR", cfg.standoff_far),
                                          ("NEAR", cfg.standoff_near)))
    cap = None
    fps = args.fps or 20.0

    if args.video:
        cap, n_frames, vid_fps = open_video(args.video)
        fps = args.fps or vid_fps

    if args.log and args.video:
        rows = read_log(args.log)
        if not rows:
            raise SystemExit(f"no rows in {args.log}")
        steps = steps_log_video(rows, cap, n_frames, args.offset)
    elif args.video:
        # No CSV to lean on, so re-run perception over the recording.
        from tello_pose import PoseEstimator, load_calibration
        K, dist, real = load_calibration(args.calib)
        if not real:
            print("!! nominal intrinsics - positions are approximate")
        steps = steps_video_only(cap, PoseEstimator(K, dist, lm.geom))
    elif args.log:
        rows = read_log(args.log)
        if not rows:
            raise SystemExit(f"no rows in {args.log}")
        steps = steps_log_only(rows)
    else:
        steps = steps_demo()

    try:
        play(steps, lm, w, h, fps=fps, save=args.save, trail_len=args.trail,
             title=f"localization map - {args.video or args.log or 'demo'}")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
