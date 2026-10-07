"""
Tello ArUco board pose estimation.

Pure perception: image in, drone-pose-in-board-frame out. No flight, no
control. tello_aruco_landing.py and tello_gui.py import from here.

Run standalone to validate the pose against a tape measure BEFORE flying:

    python tello_pose.py --live          # Tello stream, no takeoff
    python tello_pose.py --live --snap   # + press 's' to save calib frames
    python tello_pose.py --live --marker 6   # a waypoint marker instead of the board
    python tello_pose.py --image f.png   # offline, single frame
    python tello_pose.py --video f.mp4   # offline, recorded flight

FRAMES
------
Board frame (M): origin at the centre of the cross-shaped white gap,
    +X right as you face the printed side, +Y up the sheet, +Z out of the
    face.
Level frame (L): the board frame un-tilted - same origin and +X, but +Y
    is straight up (gravity) and +Z is horizontal, out into the room. For a
    vertically mounted board (tilt_deg = 0, the default) L and M are the
    same thing. Everything that flies works in L, because "up" and "level"
    have to mean what the drone's thrust means by them.
Camera frame (C): OpenCV convention, +X right, +Y down, +Z forward.
Body frame (B): +x forward, +y right, +z up.

The estimator returns the DRONE (camera) position expressed in L, which is
what a position controller needs. solvePnP natively gives the opposite
(board in camera frame), so it is inverted here once, in one place.

THE AMBIGUITY
-------------
A flat target seen from a distance has two poses that explain the image
almost equally well. They are related by a 180 deg turn about the board's
normal through the camera's line of sight: the second solution puts the
drone at (-x, -y, z) with its heading and pitch changed to match. On the
recorded flights the two fitted the image within 10% of each other in
most frames while sitting 1.2-1.6 m apart - so picking "the lower
reprojection error" was a coin toss that decided which way the drone
flew. BoardTracker breaks the tie with what else is known: the drone flies
level (so a candidate that needs a 20 deg pitch is wrong), its IMU yaw
changes smoothly, and a close, oblique view is unambiguous on its own.
When nothing can decide, the pose is marked unresolved and the controller
does not act on the lateral axis.
"""

import argparse
import math
import pathlib
import time
from dataclasses import dataclass

import cv2
import numpy as np

from tello_io import put_text

# ----------------------------------------------------------------------
# Board geometry - single source of truth, must match the printed sheet
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BoardGeometry:
    """The landing target: either the printed 2x2 board (ids 0-3, the
    default) or a single marker (ids=(one id,)), which then sits AT the
    board origin. Every tool gets it from board_config.load_board(), so a
    board.json next to the code switches them all at once."""
    dict_id: int = cv2.aruco.DICT_4X4_50
    marker_m: float = 0.070           # black edge to black edge
    pitch_m: float = 0.095            # 2x2 only: centre to centre
    ids: tuple = (0, 1, 2, 3)
    # Pad geometry, metres, measured in the level frame with a tape: the
    # board origin sits pad_drop above the pad surface, and the pad centre
    # sits pad_out out from the board origin, horizontally.
    pad_drop_m: float = 0.400
    pad_out_m: float = 0.250
    # How far the board leans back from vertical, degrees (top away from
    # the room). 0 = vertical, which is what the defaults assume.
    tilt_deg: float = 0.0

    @property
    def single(self):
        return len(self.ids) == 1

    @property
    def layout(self):
        """id -> (cx, cy) centre offset in the board plane, metres. A lone
        marker is the origin; a 2x2 board is centred on its cross-shaped gap."""
        if self.single:
            return {self.ids[0]: (0.0, 0.0)}
        if len(self.ids) != 4:
            raise ValueError(f"a board is 1 marker or 2x2 markers, not {len(self.ids)}")
        h = self.pitch_m / 2.0
        signs = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]
        return {i: (sx * h, sy * h) for i, (sx, sy) in zip(self.ids, signs)}

    @property
    def cell_m(self):
        return self.marker_m / 6.0          # 4x4 dict -> 6x6 cells with border

    @property
    def gap_m(self):
        return self.pitch_m - self.marker_m

    @property
    def span_m(self):
        """Outer size of the target: what the pose is measured across."""
        return self.marker_m if self.single else self.pitch_m + self.marker_m

    @property
    def R_level(self):
        """Board axes as columns in the level frame L. Identity when the
        board is vertical; leaning back by t turns +Z_M up into the room."""
        a = math.radians(self.tilt_deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])

    @property
    def pad_level_cm(self):
        """Pad centre in L, cm."""
        return np.array([0.0, -self.pad_drop_m, self.pad_out_m]) * 100.0

    def object_points(self):
        """id -> 4x3 corner array in board frame, ordered TL,TR,BR,BL.

        ArUco returns image corners as TL,TR,BR,BL in image order. With
        +X right and +Y up in the board plane that is:
            TL=(-h,+h) TR=(+h,+h) BR=(+h,-h) BL=(-h,-h)
        Get this order wrong and the pose is silently rotated by 90 deg.
        """
        h = self.marker_m / 2.0
        out = {}
        for mid, (cx, cy) in self.layout.items():
            out[mid] = np.array([
                [cx - h, cy + h, 0.0],
                [cx + h, cy + h, 0.0],
                [cx + h, cy - h, 0.0],
                [cx - h, cy - h, 0.0],
            ], dtype=np.float32)
        return out

    def check(self):
        if self.single:
            return (f"single marker id {self.ids[0]}, {self.marker_m*1000:.0f}mm "
                    f"cell {self.cell_m*1000:.1f}mm tilt {self.tilt_deg:.0f}deg")
        assert self.gap_m / 2 >= self.cell_m, "quiet zone too small"
        return (f"2x2 board ids {self.ids}: marker {self.marker_m*1000:.0f}mm "
                f"pitch {self.pitch_m*1000:.0f}mm gap {self.gap_m*1000:.0f}mm "
                f"cell {self.cell_m*1000:.1f}mm tilt {self.tilt_deg:.0f}deg")


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

# Nominal Tello video camera, 960x720. The 82.6 deg on the spec sheet is
# for stills; the stream is cropped, and independent calibrations of it
# land near fx = fy = 920 (about 55 x 43 deg). Good enough to smoke-test
# the pipeline, NOT good enough to fly on - every percent of focal-length
# error is a percent of range error, and the hop is flown blind on range.
NOMINAL_K = np.array([[920.0, 0.0, 480.0],
                      [0.0, 920.0, 360.0],
                      [0.0, 0.0, 1.0]])
NOMINAL_DIST = np.zeros(5)

# Plausibility bounds for a Tello calibration at 960x720. Deliberately
# wide: they catch a solve that has run away, not a slightly-off one.
HFOV_RANGE_DEG = (48.0, 75.0)


def calibration_problems(K, dist, size=(960, 720), rms=None, std=None):
    """Reasons not to trust these intrinsics; an empty list means none
    found. The checks catch what a degenerate image set produces - on the
    set this repo shipped with, all three of the first ones fired."""
    K = np.asarray(K, float)
    d = np.asarray(dist, float).ravel()
    W, H = size
    out = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if abs(fx / fy - 1.0) > 0.02:
        out.append(f"fx/fy = {fx/fy:.3f}: the Tello has square pixels, a "
                   "good calibration agrees to <1%")
    hfov = 2 * math.degrees(math.atan(W / (2 * fx)))
    if not HFOV_RANGE_DEG[0] <= hfov <= HFOV_RANGE_DEG[1]:
        out.append(f"horizontal FOV {hfov:.1f} deg is outside "
                   f"{HFOV_RANGE_DEG[0]:.0f}-{HFOV_RANGE_DEG[1]:.0f}: focal "
                   "length is not plausible for a Tello")
    if (len(d) >= 5 and abs(d[4]) > 2.0) or (len(d) >= 2 and abs(d[1]) > 1.5):
        out.append(f"distortion {np.round(d[:5], 3)} is overfitted (k2/k3 "
                   "should be small; recalibrate with k3 fixed)")
    if abs(cx - W / 2) > 0.08 * W or abs(cy - H / 2) > 0.08 * H:
        out.append(f"principal point ({cx:.0f}, {cy:.0f}) is far from the "
                   "image centre")
    if rms is not None and rms > 1.0:
        out.append(f"RMS {rms:.2f} px (> 1.0)")
    if std is not None:
        s = np.asarray(std, float).ravel()
        if len(s) >= 2 and (s[0] / fx > 0.02 or s[1] / fy > 0.02):
            out.append(f"focal length uncertain: fx {fx:.0f} +- {s[0]:.0f}, "
                       f"fy {fy:.0f} +- {s[1]:.0f} px - the views are too "
                       "alike (vary range and tilt)")
    return out


def load_calibration(path="tello_calib.npz", quiet=False):
    """Returns (K, dist, trusted). trusted is False for the nominal
    fallback AND for a file whose contents fail calibration_problems() -
    the caller decides whether that is fatal (flying) or not (viewing)."""
    try:
        d = np.load(path)
        K = d["K"].astype(np.float64)
        dist = d["dist"].astype(np.float64)
    except Exception:
        if not quiet:
            print(f"!! {path} not found - using NOMINAL intrinsics. "
                  "Pose scale will be wrong by a few percent. Do not fly on this.")
        return NOMINAL_K.copy(), NOMINAL_DIST.copy(), False
    size = tuple(int(v) for v in d["image_size"]) if "image_size" in d else (960, 720)
    rms = float(d["rms"]) if "rms" in d else None
    std = d["std_intrinsics"] if "std_intrinsics" in d else None
    problems = calibration_problems(K, dist, size, rms, std)
    # calibrate_camera.py also records what it found wrong with the views
    # themselves (too alike, too central) - a set like that can still fit
    # confidently and wrongly, so it counts too
    for p in (d["problems"] if "problems" in d else []):
        if str(p) not in problems:
            problems.append(str(p))
    if problems and not quiet:
        print(f"!! {path} looks WRONG - do not fly on it:")
        for p in problems:
            print(f"   - {p}")
        print("   Recalibrate: python calibrate_camera.py (README step 2; "
              "its docstring says what makes a calibration work).")
    return K, dist, not problems


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

@dataclass
class BoardPose:
    p_board_cm: np.ndarray      # drone (camera) position, level frame L, cm
    p_pad_cm: np.ndarray        # drone position relative to pad centre, L, cm
    yaw_deg: float              # 0 = square on, + = pointing at board's +X
    pitch_deg: float            # + = nose up (camera, relative to level)
    roll_deg: float             # + = right side down; +-180 = upside down
    bearing_px: float           # board-origin u minus principal point cx
    reproj_rms_px: float
    n_markers: int
    ids: tuple
    t_capture: float
    R_level_cam: np.ndarray = None   # camera axes (columns) in L
    t_cam_m: np.ndarray = None       # board origin in camera frame, m
    ambiguity: float = 1.0      # err(other candidate) / err(chosen); ~1 = coin toss
    degraded: bool = False      # found only in a filtered copy of a damaged frame
    resolved: bool = True       # the planar ambiguity was settled, not guessed
    source: str = "image"       # what settled it: image | gravity | heading | guess
    alt: "BoardPose" = None     # the other candidate, for diagnostics

    @property
    def range_cm(self):
        return float(self.p_board_cm[2])

    def __str__(self):
        x, y, z = self.p_board_cm
        tag = "" if self.resolved else "  AMBIGUOUS"
        return (f"board x{x:+7.1f} y{y:+7.1f} z{z:7.1f} cm | "
                f"yaw{self.yaw_deg:+6.1f} pitch{self.pitch_deg:+6.1f} | "
                f"rms{self.reproj_rms_px:5.2f}px | n{self.n_markers} "
                f"[{self.source}]{tag}")


# ----------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------

def detector_params(long_range=True):
    p = cv2.aruco.DetectorParameters()
    # Subpixel corner refinement is the single biggest accuracy win at range.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 40
    p.cornerRefinementMinAccuracy = 0.05
    if long_range:
        # At 1.5 m a 70 mm marker is only ~40 px, well under the default
        # minimum perimeter. Lower the floor and widen the threshold window
        # sweep, at the cost of some extra CPU per frame.
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 33
        p.adaptiveThreshWinSizeStep = 6
        p.polygonalApproxAccuracyRate = 0.05
    return p


def _rms(obj, img, rv, tv, K, dist):
    proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def planar_pnp_candidates(obj, img, K, dist, refine=True):
    """Both IPPE solutions of a planar target, [(rvec, tvec, rms_px)],
    lowest error first. Each is polished with VVS on its own - refining
    only the winner would bias the comparison toward it.

    If the two refine into the same pose, the image has only one minimum
    and there is no ambiguity left: a single candidate comes back. That is
    the common case with a good calibration inside ~1.5 m; with a bad one
    (or a bent sheet) the two stay apart and the choice gets hard."""
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    out = []
    for rv, tv in zip(rvecs[:n_sol], tvecs[:n_sol]):
        rv, tv = rv.copy(), tv.copy()
        if refine:
            try:
                rv, tv = cv2.solvePnPRefineVVS(obj, img, K, dist, rv, tv)
            except cv2.error:
                pass
        out.append((rv, tv, _rms(obj, img, rv, tv, K, dist)))
    out.sort(key=lambda c: c[2])
    if len(out) == 2:
        (r0, t0, _), (r1, t1, _) = out
        R0, _ = cv2.Rodrigues(r0)
        R1, _ = cv2.Rodrigues(r1)
        ang = math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        if ang < 0.5 and np.linalg.norm(t0 - t1) < 0.002:
            out = out[:1]
    return out


def solve_planar_pnp(obj, img, K, dist):
    """Planar PnP, lowest-error IPPE candidate, polished with VVS.

    A coplanar target always admits two poses (module docstring). This
    returns the better-fitting one and nothing else, which is right when
    the view is close and oblique enough to decide - room_pose.py and
    map_room.py use it that way. For the landing board, BoardTracker looks
    at both candidates instead.
    """
    c = planar_pnp_candidates(obj, img, K, dist)
    if not c:
        return None, None, None
    return c[0]


def solve_general_pnp(obj, img, K, dist):
    """Non-planar PnP, for point sets that span more than one plane (e.g.
    room markers seen across a corner). A non-coplanar configuration does
    not suffer the two-fold ambiguity solve_planar_pnp works around, so a
    single SQPnP solve plus VVS polish is enough - no ordering trick
    needed."""
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None, None
    try:
        rv, tv = cv2.solvePnPRefineVVS(obj, img, K, dist, rv, tv)
    except cv2.error:
        pass
    return rv, tv, _rms(obj, img, rv, tv, K, dist)


def is_coplanar(obj, tol_m=0.003):
    """True if obj (Nx3, metres) lies within tol_m of a single plane -
    the threshold that decides which of the two solvers above applies."""
    if len(obj) < 4:
        return True
    centered = obj - obj.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return bool(s[-1] < tol_m)


def rotation_to_ypr(R_mc):
    """Yaw/pitch/roll (deg) of camera axes R_mc (columns = camera axes
    expressed in the target/world frame, +Y up). 0 yaw = square-on (facing
    -Z), + yaw = pointing at the target's +X, + pitch = nose up, + roll =
    right side down. Roll covers +-180, so an upside-down target reads as
    ~180 instead of hiding as 0."""
    fwd = R_mc[:, 2]
    right = R_mc[:, 0]
    down = R_mc[:, 1]
    yaw = np.degrees(np.arctan2(fwd[0], -fwd[2]))
    pitch = np.degrees(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-right[1], -down[1]))
    return float(yaw), float(pitch), float(roll)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------
# Estimator: one frame in, candidate poses out
# ----------------------------------------------------------------------

class PoseEstimator:
    """Stateless per-frame board pose. estimate() returns the best-fitting
    candidate, flagged when the fit could not tell the two apart;
    BoardTracker does the temporal reasoning on top."""

    # the other candidate must fit this much worse before a single image
    # is allowed to settle the ambiguity by itself
    CONFIDENT_RATIO = 2.0

    def __init__(self, K, dist, geom=BoardGeometry(), max_rms_px=3.0,
                 long_range=True):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64)
        self.geom = geom
        self.obj = geom.object_points()
        self.max_rms = max_rms_px
        self.cx = float(self.K[0, 2])
        self.R_level = geom.R_level
        d = cv2.aruco.getPredefinedDictionary(geom.dict_id)
        self.detector = cv2.aruco.ArucoDetector(d, detector_params(long_range))

    # -- detection ----------------------------------------------------
    def detect(self, frame):
        """Markers in a frame. If the target is not among them, one more pass
        over a median-filtered copy: the Tello's Wi-Fi video loses packets,
        and the smeared blocks that leaves across a marker's black border
        are what stops it being found. On a real recording that took the
        frames with the marker found from 26% to 39%, with fewer spurious
        ids, not more (more lenient decoding got 43% - and 18x the
        spurious ids). Clean frames never pay for the second pass."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        self.last_degraded = False
        if ids is None or not any(int(i) in self.obj for i in ids.flatten()):
            c2, ids2, rej2 = self.detector.detectMarkers(cv2.medianBlur(gray, 5))
            if ids2 is not None and any(int(i) in self.obj for i in ids2.flatten()):
                # The median filter finds the marker but rounds its corners,
                # which tilts the pose: put the corners back on the real
                # image before anything measures from them.
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.05)
                c2 = tuple(cv2.cornerSubPix(gray, c.reshape(-1, 1, 2).astype(np.float32).copy(),
                                            (5, 5), (-1, -1), crit).reshape(1, 4, 2)
                           for c in c2)
                # still a damaged frame: poses from it are marked, and the
                # filter trusts them less
                self.last_degraded = True
                return c2, ids2, rej2
        return corners, ids, rejected

    def _gather(self, corners, ids):
        """Keep only markers belonging to our board. A 4X4 dictionary has a
        low Hamming distance, so filtering by known id is the main defence
        against a false positive poisoning the solve."""
        obj, img, used = [], [], []
        if ids is None:
            return None, None, ()
        seen = set()
        for c, mid in zip(corners, np.asarray(ids).flatten()):
            mid = int(mid)
            if mid not in self.obj or mid in seen:   # a duplicate id is a false hit
                continue
            seen.add(mid)
            obj.append(self.obj[mid])
            img.append(c.reshape(-1, 2))
            used.append(mid)
        if not obj:
            return None, None, ()
        return (np.concatenate(obj).astype(np.float32),
                np.concatenate(img).astype(np.float32),
                tuple(sorted(used)))

    # -- pose ---------------------------------------------------------
    def _make_pose(self, rv, tv, rms, used, t_capture):
        R_cm, _ = cv2.Rodrigues(rv)          # board -> camera
        R_mc = R_cm.T                        # camera axes in board frame
        p_m = (-R_cm.T @ tv).ravel()         # camera position in board frame, m
        R_lc = self.R_level @ R_mc           # camera axes in level frame
        p = self.R_level @ p_m               # camera position in L, m
        yaw, pitch, roll = rotation_to_ypr(R_lc)
        t = tv.ravel()
        u0 = self.K[0, 0] * t[0] / max(t[2], 1e-6) + self.K[0, 2]
        return BoardPose(
            p_board_cm=p * 100.0,
            p_pad_cm=p * 100.0 - self.geom.pad_level_cm,
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            bearing_px=float(u0 - self.cx),
            reproj_rms_px=rms, n_markers=len(used), ids=used,
            t_capture=t_capture, R_level_cam=R_lc, t_cam_m=t,
        )

    def candidates(self, frame=None, t_capture=None, detection=None):
        """All plausible poses for this frame, best fit first (0-2 of
        them). A pose whose roll says the board is upside down or on its
        side is dropped: flying on it would steer every axis backwards."""
        t_capture = time.time() if t_capture is None else t_capture
        corners, ids = (detection if detection is not None
                        else self.detect(frame)[:2])
        obj, img, used = self._gather(corners, ids)
        if obj is None or len(obj) < 4:
            return []
        raw = planar_pnp_candidates(obj, img, self.K, self.dist)
        if not raw:
            return []
        # How much worse the runner-up fits, from the raw pair - so it
        # still means something when one of them is filtered out below.
        ratio = raw[1][2] / max(raw[0][2], 1e-3) if len(raw) > 1 else 99.0
        out = []
        for rv, tv, rms in raw:
            if rms > self.max_rms:
                continue
            pose = self._make_pose(rv, tv, rms, used, t_capture)
            if abs(pose.roll_deg) > 60.0:
                continue
            pose.ambiguity = ratio
            pose.degraded = getattr(self, "last_degraded", False)
            out.append(pose)
        if len(out) == 2:
            out[0].alt, out[1].alt = out[1], out[0]
        return out

    def upside_down(self, detection):
        """True when the board is visible but held rotated - worth telling
        the operator, since the estimate is otherwise silently rejected."""
        corners, ids = detection
        obj, img, used = self._gather(corners, ids)
        if obj is None or len(obj) < 4:
            return False
        c = planar_pnp_candidates(obj, img, self.K, self.dist, refine=False)
        if not c:
            return False
        R_cm, _ = cv2.Rodrigues(c[0][0])
        return abs(rotation_to_ypr(self.R_level @ R_cm.T)[2]) > 60.0

    def estimate(self, frame=None, t_capture=None, detection=None):
        """Return the best-fitting BoardPose, or None if the board was not
        resolved. Stateless: resolved=False when the two candidates fit
        about equally well, and nothing here can say which is real.

        detection: an optional (corners, ids) already returned by detect().
        A caller that also draws the markers would otherwise pay for a
        second detectMarkers pass over the same frame, which at long-range
        parameters is the most expensive thing in the loop.
        """
        c = self.candidates(frame, t_capture, detection)
        if not c:
            return None
        best = c[0]
        best.resolved = len(c) == 1 or best.ambiguity >= self.CONFIDENT_RATIO
        best.source = "image" if best.resolved else "guess"
        return best

    # -- overlay ------------------------------------------------------
    def draw(self, frame, pose, corners=None, ids=None):
        out = frame.copy()
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        if pose is None:
            cv2.putText(out, "NO BOARD", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return out
        x, y, z = pose.p_board_cm
        px, py, pz = pose.p_pad_cm
        lines = [
            f"board  x {x:+7.1f}  y {y:+7.1f}  z {z:7.1f} cm",
            f"pad    x {px:+7.1f}  y {py:+7.1f}  z {pz:7.1f} cm",
            f"yaw {pose.yaw_deg:+6.1f}  pitch {pose.pitch_deg:+6.1f}  "
            f"roll {pose.roll_deg:+6.1f} deg",
            f"rms {pose.reproj_rms_px:.2f} px   n {pose.n_markers}   "
            f"[{pose.source}]" + ("" if pose.resolved else " AMBIGUOUS"),
        ]
        colour = (0, 255, 120) if pose.resolved else (0, 200, 255)
        for i, s in enumerate(lines):
            put_text(out, s, (12, 30 + 26 * i), 0.62, colour)
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
        return out


# ----------------------------------------------------------------------
# Tracker: which candidate is real
# ----------------------------------------------------------------------

@dataclass
class _Heading:
    """board heading in the IMU's yaw frame, beta = imu_yaw - camera_yaw.
    Constant while the board stays put (up to IMU drift), so once known it
    predicts the camera's yaw relative to the board from the IMU alone."""
    beta: float = None
    n: int = 0
    t: float = 0.0
    disagree: int = 0


class BoardTracker:
    """Chooses between the two planar-pose candidates, frame to frame.

    Evidence, strongest first:
      agree    The two are within agree_cm of each other: the drone is that
               close to the board's axis, and either answer flies the same.
               (With a single small marker this is most frames near the
               axis - the fit cannot choose, and does not need to.)
      image    Close and oblique, the perspective alone decides (the other
               candidate fits more than CONFIDENT_RATIO times worse, or
               both refine into the same pose).
      gravity  The drone flies level. The wrong candidate is the right one
               turned 180 deg about the line of sight, which tilts it by
               twice the angle the drone sits off the board's axis. A
               candidate that needs far more pitch or roll than the other
               to explain the same image is the wrong one.
      heading  Once some frames have been settled, the board's heading in
               the IMU yaw frame is known, and the IMU says which way the
               camera must be facing. The mirror candidate's heading is off
               by twice the drone's bearing around the board.
    Otherwise the best fit is returned with resolved=False: the controller
    still gets range, height and bearing (the same for both candidates) but
    no lateral error (whose sign is exactly what is in doubt).

    The heading is only ever learned from frames settled by gravity or the
    image, never from frames it settled itself, so a wrong start cannot
    reinforce itself; enough confident frames that disagree reset it.
    """

    def __init__(self, estimator, level_margin_deg=8.0, heading_margin_deg=6.0,
                 heading_commit=4, heading_max_age_s=120.0, imu_sign=None,
                 agree_cm=3.0):
        self.est = estimator
        # Candidates closer than this are the same answer for flying: the
        # mirror sits at (-x, -y, z), so they are this close only when the
        # drone is within agree_cm/2 of the board's axis either way.
        self.agree_cm = agree_cm
        self.level_margin = level_margin_deg
        self.heading_margin = heading_margin_deg
        self.heading_commit = heading_commit
        self.heading_max_age = heading_max_age_s
        # +1: IMU yaw grows clockwise, like the camera's yaw here. The
        # Tello's convention is not documented, so by default it is
        # measured (see _vote_sign) and the heading is not used until it is.
        self.imu_sign = imu_sign
        self._sign_votes = 0
        self._sign_ref = None
        self.expected_pitch = 0.0           # camera pitch in level flight
        self.h = _Heading()
        self.last = None

    def reset(self):
        self.h = _Heading()
        self.last = None

    @property
    def heading_known(self):
        return (self.imu_sign is not None and self.h.beta is not None
                and self.h.n >= self.heading_commit)

    def _vote_sign(self, yaw_vis, imu_yaw, t):
        """Settled frames a moment apart: did the IMU yaw move the same way
        as the camera's yaw relative to the board? Three consistent votes
        fix the sign; the board does not move, so only the drone turning
        changes either."""
        ref = self._sign_ref
        if ref is None:
            self._sign_ref = (t, imu_yaw, yaw_vis)
            return
        d_imu = wrap_deg(imu_yaw - ref[1])
        d_vis = wrap_deg(yaw_vis - ref[2])
        if abs(d_imu) < 4.0:
            # not turned enough to tell yet: keep accumulating from the
            # same reference (a calm approach turns slowly), unless it is
            # so old that drift would dominate
            if t - ref[0] > 20.0:
                self._sign_ref = (t, imu_yaw, yaw_vis)
            return
        self._sign_ref = (t, imu_yaw, yaw_vis)
        if abs(d_vis) < 2.0 or abs(d_imu) > 60.0:
            return
        self._sign_votes += 1 if d_imu * d_vis > 0 else -1
        if self.imu_sign is None and abs(self._sign_votes) >= 3:
            self.imu_sign = 1.0 if self._sign_votes > 0 else -1.0
            self.h = _Heading()             # learned against no sign: start over
        elif self.imu_sign is not None and self._sign_votes * self.imu_sign <= -3:
            self.imu_sign = -self.imu_sign  # a configured sign that is wrong
            self._sign_votes = 0
            self.h = _Heading()

    def _heading_pred(self, imu_yaw, t):
        if imu_yaw is None or not self.heading_known:
            return None
        if t - self.h.t > self.heading_max_age:
            self.h = _Heading()
            return None
        return wrap_deg(self.imu_sign * imu_yaw - self.h.beta)

    def _learn_heading(self, pose, imu_yaw, t):
        if imu_yaw is None:
            return
        self._vote_sign(pose.yaw_deg, imu_yaw, t)
        if self.imu_sign is None:
            return
        b = wrap_deg(self.imu_sign * imu_yaw - pose.yaw_deg)
        h = self.h
        if h.beta is None:
            self.h = _Heading(beta=b, n=1, t=t)
            return
        d = wrap_deg(b - h.beta)
        if abs(d) > 20.0:
            # A settled frame that disagrees. A few in a row mean the
            # stored heading is the wrong branch; start over from here.
            h.disagree += 1
            if h.disagree >= 3:
                self.h = _Heading(beta=b, n=1, t=t)
            return
        h.disagree = 0
        k = 1.0 / min(h.n + 1, 20)
        h.beta = wrap_deg(h.beta + k * d)
        h.n += 1
        h.t = t

    def update(self, frame=None, t_capture=None, detection=None, imu_yaw=None):
        """Best-supported BoardPose for this frame, or None."""
        t = time.time() if t_capture is None else t_capture
        cands = self.est.candidates(frame, t, detection)
        if not cands:
            return None
        best = cands[0]
        # A lone survivor: the other candidate either fit far worse
        # (image) or needed the board upside down / on its side (gravity).
        chosen = best
        source = "image" if best.ambiguity >= self.est.CONFIDENT_RATIO else "gravity"
        if len(cands) == 2:
            a, b = cands
            # gravity: how far each is from level flight
            tilt_a = abs(a.pitch_deg - self.expected_pitch) + abs(a.roll_deg)
            tilt_b = abs(b.pitch_deg - self.expected_pitch) + abs(b.roll_deg)
            pred = self._heading_pred(imu_yaw, t)
            if float(np.linalg.norm(a.p_board_cm - b.p_board_cm)) < self.agree_cm:
                # near the axis the mirror is the same place: nothing to decide
                chosen, source = a, "agree"
            elif a.ambiguity >= self.est.CONFIDENT_RATIO:
                # direct evidence first: it is also what keeps the learned
                # heading honest (a disagreeing run of these resets it)
                chosen, source = a, "image"
            elif abs(tilt_a - tilt_b) > self.level_margin:
                chosen, source = (a, "gravity") if tilt_a < tilt_b else (b, "gravity")
            elif pred is not None and abs(abs(wrap_deg(a.yaw_deg - pred))
                                          - abs(wrap_deg(b.yaw_deg - pred))) > self.heading_margin:
                chosen = min(cands, key=lambda c: abs(wrap_deg(c.yaw_deg - pred)))
                source = "heading"
            else:
                chosen, source = a, "guess"
        chosen.source = source
        chosen.resolved = source != "guess"
        if source in ("gravity", "image", "agree"):
            self._learn_heading(chosen, imu_yaw, t)
            if source == "image" and chosen.ambiguity >= 3.0:
                # Learn the level-flight pitch (camera mount, board tilt
                # error) slowly, from the frames that need no prior at all.
                self.expected_pitch += 0.05 * (chosen.pitch_deg - self.expected_pitch)
        self.last = chosen
        return chosen


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------

class PoseFilter:
    """Constant-velocity Kalman filter on the drone position (cm, level
    frame), one 2-state filter per axis, with innovation gating.

    The gating cannot latch. The old filter rejected any jump over 40 cm
    and then compared every later measurement against the value it had
    frozen - once the drone really had moved 40 cm, it rejected everything
    until the board was lost, and the controller flew on a stale number
    (up to 3/4 of the frames with the board in view, on the recorded
    flights). Here a run of consecutive rejections means the track, not
    the measurement, is wrong, and the filter restarts on the measurement.

    Measurement noise follows what the simulator measures for a 16.5 cm
    board (x2 margin for a real H.264 stream): lateral and vertical noise
    is orientation noise times range, and orientation noise itself grows
    with range, so it scales like range^4 - 0.1 cm at 0.7 m, ~1 cm at 1 m,
    ~5 cm at 1.5 m, ~15 cm at 2 m. Range (z) is about 7x steadier. The two
    axes the planar ambiguity mirrors are inflated further for unresolved
    poses.
    """

    def __init__(self, q_accel=120.0, gate_sigma=4.0, reinit_after=4,
                 max_age_s=0.6, base_sigma_cm=0.5, range4_sigma=1.0,
                 target_span_m=0.165):
        self.q = float(q_accel)          # cm/s^2, piecewise-constant acceleration
        self.gate = float(gate_sigma)
        self.reinit_after = int(reinit_after)
        self.max_age_s = float(max_age_s)
        self.base_sigma = base_sigma_cm
        # cm of lateral noise per (metre of range)^4, for the printed 16.5 cm
        # board; a smaller target is noisier by about (16.5 cm / span)^2 -
        # a single 70 mm marker ~5x (measured in the simulator)
        self.range4_sigma = range4_sigma * max((0.165 / target_span_m) ** 2, 0.5)
        self.reset()

    def reset(self):
        self.x = None                    # 3x2: per axis [position, velocity]
        self.P = None                    # 3x2x2
        self.t = None
        self.rejected = 0
        self.n_rejected = 0
        self.n_reinit = 0

    def _sigma(self, pose):
        r_m = min(max(float(np.linalg.norm(pose.p_board_cm)) / 100.0, 0.3), 3.5)
        s = self.range4_sigma * r_m ** 4
        sig = np.array([self.base_sigma + s, self.base_sigma + s,
                        0.3 + 0.15 * s])
        if not pose.resolved:
            sig[:2] *= 6.0
        if getattr(pose, "degraded", False):
            sig *= 3.0                   # from a damaged frame: keeps the track alive, lightly
        return sig

    def _predict_to(self, t):
        dt = max(0.0, t - self.t)
        if dt == 0.0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                        [dt ** 3 / 2, dt ** 2]])
        for i in range(3):
            self.x[i] = F @ self.x[i]
            self.P[i] = F @ self.P[i] @ F.T + Q
        self.t = t

    def _init(self, z, sig, t):
        self.x = np.stack([z, np.zeros(3)], axis=1)
        self.P = np.array([np.diag([s * s, 40.0 ** 2]) for s in sig])
        self.t = t
        self.rejected = 0

    def update(self, pose, now=None):
        """Feed one pose (or None) and return the filtered position at the
        pose's capture time, or None once the track is older than
        max_age_s."""
        now = time.time() if now is None else now
        if pose is None:
            return self.position() if self.fresh(now) else None
        z = np.asarray(pose.p_board_cm, float)
        t = float(pose.t_capture)
        sig = self._sigma(pose)
        if self.x is None or not self.fresh(t):
            self._init(z, sig, t)
            return self.position()
        if t < self.t:                   # out of order - ignore
            return self.position()
        self._predict_to(t)
        S = np.array([self.P[i][0, 0] + sig[i] ** 2 for i in range(3)])
        nu = z - self.x[:, 0]
        d2 = float(np.sum(nu * nu / S))
        if d2 > self.gate ** 2 * 3:
            self.rejected += 1
            self.n_rejected += 1
            if self.rejected >= self.reinit_after:
                self.n_reinit += 1
                self._init(z, sig, t)
            return self.position()
        self.rejected = 0
        for i in range(3):
            Kg = self.P[i][:, 0] / S[i]
            self.x[i] = self.x[i] + Kg * nu[i]
            self.P[i] = self.P[i] - np.outer(Kg, self.P[i][0, :])
        return self.position()

    def fresh(self, now):
        return self.x is not None and now - self.t <= self.max_age_s

    def position(self):
        return None if self.x is None else self.x[:, 0].copy()

    def velocity(self):
        return None if self.x is None else self.x[:, 1].copy()

    def predict(self, t):
        """Position extrapolated to time t (latency compensation): the
        estimate is for when the frame was captured, the command acts now."""
        if self.x is None:
            return None
        dt = float(np.clip(t - self.t, 0.0, 0.5))
        return self.x[:, 0] + self.x[:, 1] * dt


# ----------------------------------------------------------------------
# Standalone diagnostics
# ----------------------------------------------------------------------

def _run_source(est, frames, snap=False):
    """frames: iterator of BGR images. Interactive keys: q quit, s snap,
    space log a tape-measure reading."""
    tracker = BoardTracker(est)
    logged, n_snap = [], 0
    warned = False
    for frame in frames:
        if frame is None:
            continue
        corners, ids, _ = est.detect(frame)
        pose = tracker.update(frame, detection=(corners, ids))
        if pose is None and not warned and est.upside_down((corners, ids)):
            print("!! the board is upside down or on its side - "
                  "the pose would steer every axis backwards")
            warned = True
        view = est.draw(frame, pose, corners, ids)
        cv2.imshow("tello pose", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and snap:
            d = pathlib.Path("calib")
            d.mkdir(exist_ok=True)
            fn = d / f"calib_{n_snap:03d}.png"
            while fn.exists():
                n_snap += 1
                fn = d / f"calib_{n_snap:03d}.png"
            cv2.imwrite(str(fn), frame)
            n_snap += 1
            print(f"saved {fn}")
        if k == ord(" ") and pose is not None:
            logged.append(pose)
            print(f"[{len(logged):2d}] {pose}")
    cv2.destroyAllWindows()
    if logged:
        print("\ntape-measure log (compare z against your ruler):")
        for j, p in enumerate(logged, 1):
            x, y, z = p.p_board_cm
            print(f"{j:3d}  x {x:+7.1f}  y {y:+7.1f}  z {z:7.1f}  "
                  f"yaw {p.yaw_deg:+6.1f}  rms {p.reproj_rms_px:4.2f}  "
                  f"{'' if p.resolved else 'AMBIGUOUS'}")


def main():
    from tello_io import file_frames, open_drone, tello_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Tello stream, no takeoff")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--calib", default="tello_calib.npz")
    ap.add_argument("--snap", action="store_true", help="'s' saves calib frames")
    ap.add_argument("--marker", type=int, metavar="ID",
                    help="localize this waypoint marker instead of the board")
    ap.add_argument("--marker-mm", type=float, default=150.0,
                    help="waypoint marker size, black edge to edge (default 150)")
    args = ap.parse_args()

    from board_config import load_board
    geom = load_board()
    if args.marker is not None:
        # a waypoint marker, located in its own frame (origin at its centre)
        geom = BoardGeometry(dict_id=geom.dict_id, marker_m=args.marker_mm / 1000.0,
                             ids=(args.marker,), pad_drop_m=1.0, pad_out_m=0.0)
    print(geom.check())
    K, dist, trusted = load_calibration(args.calib)
    est = PoseEstimator(K, dist, geom)

    if args.image:
        img = cv2.imread(args.image)
        corners, ids, _ = est.detect(img)
        pose = est.estimate(img, detection=(corners, ids))
        print(pose if pose else "no board")
        if pose is not None and pose.alt is not None:
            print(f"  other candidate: {pose.alt}")
        cv2.imshow("tello pose", est.draw(img, pose, corners, ids))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.video:
        _run_source(est, file_frames(args.video))
    elif args.live:
        print("keys: q quit | space log reading | s save calib frame")
        drone = open_drone()
        print(f"battery {drone.get_battery()}%  -- NOT taking off")
        try:
            _run_source(est, tello_frames(drone), snap=args.snap)
        finally:
            drone.streamoff()
            drone.end()
    else:
        ap.error("pick one of --live, --image, --video")


if __name__ == "__main__":
    main()
