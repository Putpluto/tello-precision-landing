"""
A synthetic Tello for the tests: printed ArUco sheets standing in a room,
seen through the nominal camera, a drone that turns rc commands into
velocities, video that arrives late, and a fake clock. Enough to fly
tello_aruco_landing.Lander end to end with no hardware. It is not a model
of the Tello's dynamics - nothing was tuned on it - so a test passing here
says the code is wired right, not that a real flight will be.

World frame: +Y up, floor at y = 0, centimetres. Yaw is clockwise seen
from above, like the drone's: forward = (-sin yaw, 0, cos yaw).
"""

import bisect
import math
from types import SimpleNamespace

import cv2
import numpy as np

from landing_control import BOARD, Obs
from tello_io import PLACEHOLDER_SHAPE
from tello_pose import NOMINAL_K

UP = np.array([0.0, 1.0, 0.0])
SIZE = (960, 720)
PX_PER_CM = 30                      # sheet artwork resolution


def heading(yaw_deg):
    """(forward, right), level unit vectors in the world."""
    a = math.radians(yaw_deg)
    return (np.array([-math.sin(a), 0.0, math.cos(a)]),
            np.array([-math.cos(a), 0.0, -math.sin(a)]))


def camera_axes(yaw_deg):
    """Camera axes as columns in the world: OpenCV's right, down, forward."""
    f, r = heading(yaw_deg)
    return np.stack([r, -UP, f], axis=1)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------
# Printed sheets
# ----------------------------------------------------------------------

def marker_art(dict_id, marker_id, marker_cm=15.0, margin_cm=2.5):
    side = int(round(marker_cm * PX_PER_CM))
    m = int(round(margin_cm * PX_PER_CM))
    img = np.full((side + 2 * m, side + 2 * m), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img[m:m + side, m:m + side] = cv2.aruco.generateImageMarker(d, int(marker_id), side)
    return img


def board_art(geom, margin_cm=2.0):
    """The landing target as printed: geom's markers about its origin."""
    side = int(round(geom.marker_m * 100 * PX_PER_CM))
    w = int(round((geom.span_m * 100 + 2 * margin_cm) * PX_PER_CM))
    img = np.full((w, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(geom.dict_id)
    for mid, (cx, cy) in geom.layout.items():
        u = int(round(w / 2 + cx * 100 * PX_PER_CM - side / 2))
        v = int(round(w / 2 - cy * 100 * PX_PER_CM - side / 2))
        img[v:v + side, u:u + side] = cv2.aruco.generateImageMarker(d, int(mid), side)
    return img


class Sheet:
    """A sheet standing upright, centre at `centre` (cm), printed face
    toward `normal` (horizontal). Its frame is the target frame tello_pose
    measures in: +X right as you face the print, +Y up, +Z out of the face."""

    def __init__(self, art, centre, normal):
        self.levels = [art]                 # mipmaps, so a far sheet is not aliased
        while min(self.levels[-1].shape) > 16:
            a = self.levels[-1]
            self.levels.append(cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2),
                                          interpolation=cv2.INTER_AREA))
        self.size_cm = (art.shape[1] / PX_PER_CM, art.shape[0] / PX_PER_CM)
        self.centre = np.asarray(centre, float)
        n = np.asarray(normal, float)
        self.normal = n / np.linalg.norm(n)

    @property
    def axes(self):
        """Sheet axes (columns x, y, z) in the world."""
        return np.stack([np.cross(-self.normal, UP), UP, self.normal], axis=1)

    def corners(self):
        """TL, TR, BR, BL in the world, cm."""
        w, h = self.size_cm
        x, y = self.axes[:, 0] * w / 2, self.axes[:, 1] * h / 2
        c = self.centre
        return np.array([c - x + y, c + x + y, c + x - y, c - x - y])

    def to_local(self, pos_cm, yaw_deg):
        """Camera position and camera axes in this sheet's frame: what
        tello_pose should measure from here."""
        R = self.axes
        return R.T @ (np.asarray(pos_cm, float) - self.centre), R.T @ camera_axes(yaw_deg)

    def in_view(self, pos_cm, yaw_deg, half_h_deg=26.0, half_v_deg=20.0, max_cm=350.0,
                max_oblique_deg=60.0):
        """Whether the camera would see (and read) it - for tests that skip
        the images."""
        d = self.centre - np.asarray(pos_cm, float)
        if -d @ self.normal < math.cos(math.radians(max_oblique_deg)) * np.linalg.norm(d):
            return False
        c = camera_axes(yaw_deg).T @ d
        return (0 < c[2] < max_cm and abs(c[0]) < c[2] * math.tan(math.radians(half_h_deg))
                and abs(c[1]) < c[2] * math.tan(math.radians(half_v_deg)))


class Scene:
    """Sheets in a room, rendered as the Tello's camera would see them."""

    def __init__(self, sheets, K=NOMINAL_K, size=SIZE, background=118, noise=2.0, seed=1):
        self.sheets = list(sheets)
        self.K = np.asarray(K, float)
        self.size = size
        self.background = background
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def render(self, pos_cm, yaw_deg):
        W, H = self.size
        frame = np.full((H, W), self.background, np.uint8)
        C = np.asarray(pos_cm, float)
        Rc = camera_axes(yaw_deg)
        for s in sorted(self.sheets, key=lambda s: -np.linalg.norm(s.centre - C)):
            if (C - s.centre) @ s.normal <= 0:          # its back
                continue
            cam = (Rc.T @ (s.corners() - C).T).T
            if (cam[:, 2] < 5.0).any():
                continue
            uv = (cam[:, :2] / cam[:, 2:3]) * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
            if (uv[:, 0].max() < 0 or uv[:, 0].min() > W or uv[:, 1].max() < 0
                    or uv[:, 1].min() > H):
                continue
            width_px = np.linalg.norm(uv[1] - uv[0])
            k = int(np.clip(math.floor(math.log2(max(
                s.levels[0].shape[1] / max(width_px, 1.0), 1.0))), 0, len(s.levels) - 1))
            art = s.levels[k]
            h, w = art.shape
            src = np.float32([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
            M = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
            cv2.warpPerspective(art, M, (W, H), dst=frame, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
        frame = cv2.GaussianBlur(frame, (3, 3), 0.6)
        if self.noise:
            frame = np.clip(frame + self.rng.normal(0, self.noise, frame.shape), 0,
                            255).astype(np.uint8)
        return frame


def route_scene(geom, dict_id=None):
    """The route the tests fly, cm. From each stop - 100 cm in front of a
    marker, level with it - the next target is a quarter turn away:
    right, right, right, right, then LEFT for the board, 60 cm lower.

        returns {target: Sheet}, pad centre (world, cm)
    """
    dict_id = geom.dict_id if dict_id is None else dict_id
    place = {4: ((0, 100, 250), (0, 0, -1)),
             5: ((-250, 100, 150), (1, 0, 0)),
             6: ((-150, 100, -100), (0, 0, 1)),
             7: ((100, 100, 0), (-1, 0, 0))}
    sheets = {mid: Sheet(marker_art(dict_id, mid), c, n) for mid, (c, n) in place.items()}
    board = Sheet(board_art(geom), (20, 40, -250), (0, 0, 1))
    sheets[BOARD] = board
    pad = board.centre + board.normal * geom.pad_out_m * 100 - UP * geom.pad_drop_m * 100
    return sheets, pad


# ----------------------------------------------------------------------
# The drone
# ----------------------------------------------------------------------

class Body:
    """Kinematics: rc a/b/c (right/forward/up) set a body-frame velocity
    the drone reaches with a first-order lag; d turns it clockwise. `pos`
    is the camera's; the airframe centre is cam_fwd behind it."""

    def __init__(self, pos=(0, 0, 0), yaw=0.0, flying=False, rc_cm_s=1.0, rc_deg_s=1.0,
                 tau_s=0.3, cam_fwd=4.0):
        self.pos = np.array(pos, float)
        self.yaw = float(yaw)
        self.v_body = np.zeros(3)          # forward, right, up
        self.rc = (0, 0, 0, 0)
        self.flying = flying
        self.rc_cm_s, self.rc_deg_s, self.tau = rc_cm_s, rc_deg_s, tau_s
        self.cam_fwd = cam_fwd
        self.move = None                   # (world displacement still to go, cm/s): go, land

    def centre(self):
        """The airframe centre: what has to end up over the pad."""
        return self.pos - heading(self.yaw)[0] * self.cam_fwd

    def v_world(self):
        f, r = heading(self.yaw)
        return f * self.v_body[0] + r * self.v_body[1] + UP * self.v_body[2]

    def advance(self, dt):
        if self.move is not None:          # a go or a landing: straight there, at speed
            rem, speed = self.move
            d = float(np.linalg.norm(rem))
            step = rem if d <= speed * dt else rem * (speed * dt / d)
            self.pos += step
            self.move = None if d <= speed * dt else (rem - step, speed)
            self.v_body[:] = 0.0
            return
        if not self.flying:
            self.v_body[:] = 0.0
            return
        a, b, c, d = self.rc
        cmd = np.array([b, a, c], float) * self.rc_cm_s
        self.v_body += (cmd - self.v_body) * min(dt / self.tau, 1.0)
        self.pos += self.v_world() * dt
        self.yaw = wrap(self.yaw + d * self.rc_deg_s * dt)

    def displacement(self, fwd, left, up=0.0):
        f, r = heading(self.yaw)
        return f * fwd - r * left + UP * up

    def go(self, fwd, left, up=0.0):
        """Jump there at once (FakeTello flies it, at speed)."""
        self.pos += self.displacement(fwd, left, up)
        self.v_body[:] = 0.0


def true_obs(body, sheet, t, battery=80, resolved=True):
    """landing_control.Obs straight from the geometry, no images: what a
    perfect localizer would report when the sheet is in view."""
    if not sheet.in_view(body.pos, body.yaw):
        return Obs(t=t, battery=battery)
    p, R = sheet.to_local(body.pos, body.yaw)
    v = sheet.axes.T @ body.v_world()
    return Obs(t=t, p=p, v=v, pose=SimpleNamespace(R_level_cam=R, resolved=resolved),
               battery=battery)


class FakeTello:
    """The parts of djitellopy.Tello that Lander uses, over a Scene. Its
    clock is the test's: pass now/sleep to Lander, and sleeping is what
    moves the drone. Frames come out RGB (as djitellopy's do), late by
    video_delay_s, starting with djitellopy's black placeholder."""

    def __init__(self, scene, pos=(0, 0, 0), yaw=0.0, video_delay_s=0.25, fps=30.0,
                 battery=80, takeoff_cm=80.0, imu_yaw_offset=23.0, **body_kw):
        self.scene = scene
        self.body = Body(pos, yaw, **body_kw)
        self.delay, self.fps = video_delay_s, fps
        self.battery = battery
        self.takeoff_cm = takeoff_cm
        self.imu_yaw_offset = imu_yaw_offset
        self.t = 0.0
        self._times = [0.0]
        self._poses = [(self.body.pos.copy(), self.body.yaw)]
        self._frame = np.zeros(PLACEHOLDER_SHAPE, np.uint8)
        self._frame_t = None
        self.landed_at = None
        self.log = []                       # (t, command)

    # -- clock --------------------------------------------------------------
    def now(self):
        return self.t

    def sleep(self, dt):
        n = max(1, math.ceil(dt * 60.0))
        for _ in range(n):
            self.body.advance(dt / n)
            self.t += dt / n
            self._times.append(self.t)
            self._poses.append((self.body.pos.copy(), self.body.yaw))
        if len(self._times) > 600:          # keep ~10 s
            del self._times[:-300], self._poses[:-300]

    def _pose_at(self, t):
        i = max(bisect.bisect_right(self._times, t) - 1, 0)
        return self._poses[i]

    # -- video ----------------------------------------------------------------
    @property
    def frame(self):
        if self.t < 0.3:
            return self._frame              # still the placeholder
        if self._frame_t is None or self.t - self._frame_t >= 1.0 / self.fps - 1e-9:
            pos, yaw = self._pose_at(self.t - self.delay)
            self._frame = cv2.cvtColor(self.scene.render(pos, yaw), cv2.COLOR_GRAY2RGB)
            self._frame_t = self.t
        return self._frame

    def get_frame_read(self):
        return self                         # .frame, like BackgroundFrameRead

    # -- commands -------------------------------------------------------------
    @property
    def is_flying(self):
        return self.body.flying

    def get_battery(self):
        return self.battery

    def get_current_state(self):
        b = self.body
        return {"bat": self.battery, "yaw": int(round(wrap(b.yaw + self.imu_yaw_offset))),
                "pitch": 0, "roll": 0, "vgx": int(round(b.v_body[0])),
                "vgy": int(round(b.v_body[1])), "vgz": int(round(-b.v_body[2])),
                "h": int(round(b.pos[1])), "tof": int(round(b.pos[1])) + 10}

    def send_rc_control(self, a, b, c, d):
        self.body.rc = tuple(int(np.clip(v, -100, 100)) for v in (a, b, c, d))

    def takeoff(self):
        self.log.append((self.t, "takeoff"))
        self.body.flying = True
        self.body.pos[1] = self.takeoff_cm
        self.sleep(2.0)

    def go_xyz_speed(self, x, y, z, speed):
        self.log.append((self.t, f"go {x} {y} {z} {speed}"))
        if not all(-500 <= v <= 500 for v in (x, y, z)) or max(abs(x), abs(y), abs(z)) < 20:
            raise RuntimeError("out of range")   # the Tello's 'error' reply
        self.body.rc = (0, 0, 0, 0)
        self.body.move = (self.body.displacement(x, y, z), float(speed))
        self.sleep(math.sqrt(x * x + y * y + z * z) / speed + 0.5)

    def land(self):
        self.log.append((self.t, "land"))
        self.body.rc = (0, 0, 0, 0)
        drop = 0.0
        if self.body.flying:
            c = self.body.centre()
            self.landed_at = np.array([c[0], 0.0, c[2]])
            drop = max(self.body.pos[1] - 5.0, 0.0)     # the lens ends ~5 cm up
            self.body.move = (-UP * drop, 40.0)
        self.body.flying = False
        self.sleep(max(2.0, drop / 40.0 + 0.5))

    def emergency(self):
        self.log.append((self.t, "emergency"))
        self.body.flying = False

    def streamoff(self):
        pass

    def end(self):
        pass
