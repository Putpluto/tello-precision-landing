"""
Visit waypoint markers, then search for the 4-marker ArUco board and land
on the pad in front of it.

    python tello_aruco_landing.py --view                    # marker 4, right to 5, left to the board, land
    python tello_aruco_landing.py --view --waypoints        # straight to the board and land
    python tello_aruco_landing.py --view --hold             # stop 70 cm from the board, don't land
    python tello_aruco_landing.py --view --log run.csv

    takeoff
      for each waypoint marker (in the order given):
        -> SEARCH    turn on the spot until that marker is seen
        -> GOTO      fly to 100 cm in front of it and settle
        -> HOVER     hold there for 2 s
      -> SEARCH    turn on the spot until the board is seen
      -> APPROACH  fly to 150 cm in front of the board and settle
      -> CLOSE     fly to 70 cm in front of it and settle
      -> HOP       one 'go' forward over the pad (the camera can't see it)
      -> LAND

Each search only turns on the spot, so the next marker (or the board) must
be visible from where the drone stops at the previous one.

Localization is tello_pose.py's, the same as the GUI: both mirror-image
poses of the target (PoseEstimator), the real one picked by BoardTracker
(image fit, gravity, IMU heading), smoothed by PoseFilter (Kalman). The
board is whatever board_config.py has configured (default: the 2x2 board).

Keys (with --view): q = land now, x = cut the motors (the drone DROPS).

Safety: refuses to fly on a bad calibration, lands if the battery drops
below 15%, if the mission runs too long, if a marker isn't found within a
full turn, or if the hop it computes is implausible - and always lands on
exit. There is no obstacle sensing: keep the paths clear.
"""

import argparse
import csv
import math
import threading
import time

import cv2
import numpy as np

from board_config import load_board
from tello_io import FrameGrabber, open_drone
from tello_map import LocalizationMap
from tello_pose import (BoardGeometry, BoardTracker, PoseEstimator, PoseFilter,
                        load_calibration)

# ----------------------------------------------------------------------
# Settings (cm, degrees, seconds)
# ----------------------------------------------------------------------

# The route: marker 4, marker 5, then the board. While searching for each
# target the drone turns this way on the spot: +1 = right (clockwise),
# -1 = left. Find 4 (turning right), turn right to find 5, turn left to
# find the board.
WAYPOINTS = (4, 5)
SEARCH_TURN = {4: +1, 5: +1, "board": -1}

WAYPOINT_CM = 100.0         # stop this far in front of each waypoint marker
WAYPOINT_HOVER_S = 2.0      # and hover there this long
FAR_CM = 150.0              # first stop in front of the board
NEAR_CM = 70.0              # last stop before the hop
HEIGHT_CM = 0.0             # fly level with the centre of the marker / board
RATE_HZ = 15.0
VIDEO_DELAY_S = 0.25        # the video lags; positions are predicted across it

# settled = every error (cm, deg) inside these, moving slower than speed (cm/s), for SETTLE_S
TOL_WAYPOINT = dict(fwd=15, lat=12, up=15, yaw=8, speed=20)
TOL_FAR = dict(fwd=15, lat=12, up=15, yaw=8, speed=20)
TOL_NEAR = dict(fwd=5, lat=3.5, up=6, yaw=4, speed=7)
SETTLE_S = 0.8

SEARCH_RC = 22              # how fast to turn while searching
SEARCH_MAX_S = 25.0         # a bit more than one full turn
LOST_S = 1.2                # target out of view this long -> search again
CLOSE_MAX_S = 40.0          # can't settle at NEAR -> land where it is
GOTO_MAX_S = 40.0           # can't settle at a waypoint -> land where it is
MISSION_MAX_S = 120.0       # plus WAYPOINT_EXTRA_S per waypoint
WAYPOINT_EXTRA_S = 45.0
BATTERY_MIN = 15
AFTER_LAND_S = 5.0          # keep recording this long after touching down
HOP_SPEED = 30              # cm/s
HOP_MAX_CM = 95.0
CAMERA_AHEAD_CM = 4.0       # the lens sits this far in front of the drone's centre
WAYPOINT_MARKER_M = 0.150   # print/room_marker_*_A4.pdf
MAP_SIZE = (564, 720)       # the map beside the 960x720 camera view

# per axis: kp (rc per cm, per deg for yaw), kd (rc per cm/s), ki (rc per cm*s), limit (rc)
GAINS = dict(fwd=(0.55, 0.30, 0.10, 30), lat=(0.55, 0.30, 0.10, 25),
             up=(0.60, 0.20, 0.10, 25), yaw=(0.90, 0.00, 0.00, 30))
SLEW = 10                   # max rc change per step: the Tello dislikes jumps


# ----------------------------------------------------------------------
# Control
# ----------------------------------------------------------------------

class Axis:
    """PD (+ a small I to cancel steady drift once close) for one rc channel."""

    def __init__(self, kp, kd, ki, limit):
        self.kp, self.kd, self.ki, self.limit = kp, kd, ki, limit
        self.i = 0.0
        self.out = 0.0

    def step(self, err, vel, dt):
        # vel is along the same axis as err: moving toward the target makes
        # err*vel > 0. Integrate only near the target and when not closing in.
        if self.ki and abs(err) < 30 and err * vel <= 2 * abs(err):
            self.i = float(np.clip(self.i + self.ki * err * dt, -12, 12))
        u = np.clip(self.kp * err - self.kd * vel + self.i, -self.limit, self.limit)
        self.out = float(np.clip(u, self.out - SLEW, self.out + SLEW))
        return self.out

    def reset(self):
        self.i = self.out = 0.0


def body_axes(R):
    """The drone's forward and right directions, flattened onto the floor,
    from the camera axes R (target frame)."""
    fwd, right = R[:, 2].copy(), R[:, 0].copy()
    fwd[1] = right[1] = 0.0
    return fwd / np.linalg.norm(fwd), right / np.linalg.norm(right)


class Controller:
    """Fly to `target` with the nose pointed at the centre of the marker/board."""

    def __init__(self):
        self.axes = {k: Axis(*g) for k, g in GAINS.items()}

    def reset(self):
        for a in self.axes.values():
            a.reset()

    def step(self, p, v, R, target, use_lateral, dt):
        fwd, right = body_axes(R)
        e, aim = target - p, np.array([0.0, HEIGHT_CM, 0.0]) - p
        err = dict(fwd=e @ fwd, lat=e @ right, up=e[1],
                   yaw=math.degrees(math.atan2(aim @ right, aim @ fwd)))
        vel = dict(fwd=v @ fwd, lat=v @ right, up=v[1])
        if use_lateral:
            a = self.axes["lat"].step(err["lat"], vel["lat"], dt)
        else:
            # mirror pose possible: the sign of the sideways error is unknown
            self.axes["lat"].i = 0.0
            a = self.axes["lat"].step(0.0, 0.0, dt)
        b = self.axes["fwd"].step(err["fwd"], vel["fwd"], dt)
        c = self.axes["up"].step(err["up"], vel["up"], dt)
        d = self.axes["yaw"].step(err["yaw"], 0.0, dt)
        err["speed"] = float(np.linalg.norm(v if use_lateral else v * [0, 1, 1]))
        return tuple(int(round(x)) for x in (a, b, c, d)), err


def hop_vector(p, R, pad_out_cm):
    """(forward, right) cm the drone's centre must move to be over the pad."""
    fwd, right = body_axes(R)
    centre = p - fwd * CAMERA_AHEAD_CM
    d = np.array([0.0, centre[1], pad_out_cm]) - centre
    return float(d @ fwd), float(d @ right)


# ----------------------------------------------------------------------
# Localization of one target (tello_pose.py)
# ----------------------------------------------------------------------

class Perception:
    """The board, or one waypoint marker: estimator, tracker, Kalman
    filter and map, each for that target's geometry."""

    def __init__(self, K, dist, geom, standoffs):
        self.geom = geom
        self.est = PoseEstimator(K, dist, geom)
        self.tracker = BoardTracker(self.est)
        self.filt = PoseFilter(target_span_m=geom.span_m)
        self.lmap = LocalizationMap(geom, standoffs=standoffs)
        self.trail = []


def waypoint_geometry(board, marker_id, marker_m=WAYPOINT_MARKER_M):
    """A single waypoint marker, in its own frame (origin at its centre).
    The pad fields only place the floor line on its map: the marker is
    ~1 m up and nothing lands there."""
    return BoardGeometry(dict_id=board.dict_id, marker_m=marker_m, ids=(int(marker_id),),
                         pad_drop_m=1.0, pad_out_m=0.0)


# ----------------------------------------------------------------------
# The mission
# ----------------------------------------------------------------------

class Lander:
    """The whole mission. The command line runs it with an OpenCV window
    (view=True); tello_gui.py runs it in a thread, gets each rendered view
    through on_frame and each message through on_note, asks it to stop with
    the `stop` Event, and keeps the connection afterwards (disconnect=False)."""

    def __init__(self, drone, K, dist, waypoints=(), hold=False, view=False, log=None,
                 record=None, on_frame=None, on_note=None, stop=None, disconnect=True,
                 airborne=None, board=None, waypoint_marker_m=WAYPOINT_MARKER_M,
                 now=time.monotonic, sleep=time.sleep):
        self.drone = drone
        board = board or load_board()
        self.pad_out_cm = board.pad_out_m * 100.0
        self.perc = {"board": Perception(K, dist, board, (("FAR", FAR_CM), ("NEAR", NEAR_CM)))}
        for w in waypoints:
            self.perc[int(w)] = Perception(K, dist, waypoint_geometry(board, w, waypoint_marker_m),
                                           (("STOP", WAYPOINT_CM),))
        # None: ask djitellopy. The GUI says, since it takes off with raw commands.
        self.airborne_at_start = airborne
        self.on_frame, self.on_note = on_frame, on_note
        self.stop, self.disconnect = stop, disconnect
        self.targets = [int(w) for w in waypoints] + ["board"]
        self.hold, self.view = hold, view
        self.now, self.sleep = now, sleep
        self.max_s = MISSION_MAX_S + WAYPOINT_EXTRA_S * len(waypoints)
        self.frames = FrameGrabber(drone)
        self.ctrl = Controller()
        self.pose = None                     # latest pose (orientation, resolved)
        self.det = None
        self.frame = None
        self.log = None
        if log:
            self.logf = open(log, "w", newline="")
            self.log = csv.writer(self.logf)
            self.log.writerow(["t", "target", "state", "x", "y", "z", "vx", "vy", "vz",
                               "yaw", "resolved", "rc_right", "rc_fwd", "rc_up", "rc_yaw",
                               "battery", "note"])
        self.record = record
        self.video = None                    # opened in run(): what the window shows, as .mp4
        self.reason = ""
        self.warned_upside_down = set()

    @property
    def target(self):
        return self.targets[0]

    def say(self, msg):
        line = f"[{self.now() - self.t0:6.1f}s] {msg}"
        print(line)
        self.note = msg
        if self.on_note:
            self.on_note(line)

    def goto(self, state, msg=None):
        self.say(msg or f"-> {state}")
        self.state = state
        self.t_state = self.now()
        self.t_settled = None
        self.ctrl.reset()

    def next_target(self):
        """Done with this waypoint: forget its track and search for the next one."""
        done = self.targets.pop(0)
        self.pose = None
        self.turn = SEARCH_TURN.get(self.target, 1)
        name = "the board" if self.target == "board" else f"marker {self.target}"
        side = "right" if self.turn > 0 else "left"
        self.goto("SEARCH", f"marker {done} done -> turn {side}, SEARCH for {name}")

    def settled(self, err, tol, scale=1.0):
        ok = all(abs(err[k]) < tol[k] * scale for k in tol)
        if not ok:
            self.t_settled = None
            return False
        if self.t_settled is None:
            self.t_settled = self.now()
        return self.now() - self.t_settled >= SETTLE_S

    # -- one control step -------------------------------------------------
    def observe(self):
        """New frame -> pose of the current target -> track. Returns the
        position predicted to now, or None if the target hasn't been seen
        recently."""
        now = self.now()
        perc = self.perc[self.target]
        frame, new = self.frames.grab()
        if new:
            self.frame = frame
            self.det = perc.est.detect(frame)[:2]
            t_cap = now - VIDEO_DELAY_S
            pose = perc.tracker.update(detection=self.det, t_capture=t_cap,
                                       imu_yaw=self.imu_yaw)
            if pose is not None:
                self.pose = pose
                perc.filt.update(pose, now=t_cap)
                p = perc.filt.position()
                if p is not None:
                    perc.trail = (perc.trail + [tuple(p)])[-300:]
            elif (perc.est.upside_down(self.det)
                  and self.target not in self.warned_upside_down):
                self.warned_upside_down.add(self.target)
                what = "BOARD" if self.target == "board" else f"MARKER {self.target}"
                self.say(f"!! THE {what} IS UPSIDE DOWN - mount it with the TOP mark up"
                         + (" (id 0 top left)" if self.target == "board" else "")
                         + ". Ignoring it until then.")
        if self.pose is not None and perc.filt.fresh(now - VIDEO_DELAY_S):
            self.v = perc.filt.velocity()
            return perc.filt.predict(now)
        return None

    def step(self, p, battery, dt):
        """The state machine. Returns (rc, action) - action None, 'hop' or 'land'."""
        now = self.now()
        if now - self.t0 > self.max_s:
            return self.finish("mission took too long")
        if battery is not None and battery < BATTERY_MIN:
            return self.finish(f"battery {battery}%")
        seen = p is not None
        if seen:
            self.t_seen = now

        if self.state == "SEARCH":
            if seen:
                if self.target == "board":
                    self.goto("APPROACH", "board found -> APPROACH")
                else:
                    self.goto("GOTO", f"marker {self.target} found -> GOTO")
                return (0, 0, 0, 0), None
            if now - self.t_state > SEARCH_MAX_S:
                what = "board" if self.target == "board" else f"marker {self.target}"
                return self.finish(f"{what} not found")
            return (0, 0, 0, self.turn * SEARCH_RC), None

        if self.state in ("GOTO", "HOVER", "APPROACH", "CLOSE", "HOLD"):
            if not seen:
                if self.state == "HOVER" and now - self.t_state >= WAYPOINT_HOVER_S:
                    self.next_target()
                elif now - self.t_seen > LOST_S and self.state != "HOVER":
                    self.goto("SEARCH", "lost it -> SEARCH")
                return (0, 0, 0, 0), None
            stand = {"GOTO": WAYPOINT_CM, "HOVER": WAYPOINT_CM,
                     "APPROACH": FAR_CM}.get(self.state, NEAR_CM)
            target = np.array([0.0, HEIGHT_CM, stand])
            rc, err = self.ctrl.step(p, self.v, self.pose.R_level_cam, target,
                                     self.pose.resolved, dt)
            self.err = err
            if abs(err["yaw"]) > 2:
                self.turn = 1 if err["yaw"] > 0 else -1    # if lost, turn this way
            in_state = now - self.t_state
            if self.state == "GOTO":
                if self.settled(err, TOL_WAYPOINT):
                    self.goto("HOVER", f"at marker {self.target} -> HOVER "
                                       f"{WAYPOINT_HOVER_S:.0f} s")
                elif in_state > GOTO_MAX_S:
                    return self.finish(f"could not settle at marker {self.target}")
            elif self.state == "HOVER":
                if in_state >= WAYPOINT_HOVER_S:
                    self.next_target()
                    return (0, 0, 0, 0), None
            elif self.state == "APPROACH" and self.settled(err, TOL_FAR):
                self.goto("CLOSE", "settled at FAR -> CLOSE")
            elif self.state == "CLOSE":
                # after 12 s, loosen the tolerances a little (up to 1.6x)
                scale = 1 + min(max(in_state - 12, 0) / 10, 0.6)
                if self.settled(err, TOL_NEAR, scale):
                    self.goto("HOLD" if self.hold else "HOP",
                              "settled at NEAR -> " + ("HOLD" if self.hold else "HOP"))
                elif in_state > CLOSE_MAX_S:
                    return self.finish("could not settle at NEAR")
            return rc, None

        if self.state == "HOP":
            if not seen:
                return self.finish("board lost just before the hop")
            fwd, right = hop_vector(p, self.pose.R_level_cam, self.pad_out_cm)
            if not self.pose.resolved:
                right = 0.0     # side unknown, but it settled within 3.5 cm of the axis
            if not (-5 <= fwd <= HOP_MAX_CM and abs(right) <= 40):
                return self.finish(f"hop ({fwd:.0f}, {right:.0f}) cm looks wrong")
            self.reason = "landed on the pad"
            self.state = "LAND"
            if max(abs(fwd), abs(right)) < 20:          # 'go' won't do under 20 cm
                self.say(f"already over the pad ({fwd:.0f}, {right:.0f}) cm - land")
                return (0, 0, 0, 0), "land"
            self.say(f"HOP forward {fwd:.0f} cm, right {right:.0f} cm")
            self.hop = (int(round(fwd)), int(round(-right)))   # go_xyz: x fwd, y left
            return (0, 0, 0, 0), "hop"
        return (0, 0, 0, 0), None

    def finish(self, why):
        self.reason = why
        self.say(f"LAND: {why}")
        self.state = "LAND"
        return (0, 0, 0, 0), "land"

    # -- the loop ---------------------------------------------------------
    def run(self):
        drone = self.drone
        self.t0 = self.t_state = self.t_seen = self.now()
        self.state, self.t_settled, self.note, self.err = "SEARCH", None, "", None
        self.turn = SEARCH_TURN.get(self.target, 1)
        self.v, self.imu_yaw = np.zeros(3), None
        self.rc, self.battery = (0, 0, 0, 0), drone.get_battery()
        self.say(f"battery {self.battery}%   route: " + " -> ".join(
            "board" if t == "board" else f"marker {t}" for t in self.targets))
        if self.record:
            self.video = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                         RATE_HZ, (960 + MAP_SIZE[0], 720))
            if not self.video.isOpened():
                raise SystemExit(f"cannot write {self.record}")
        # The display/recorder runs on its own clock, so the hop and the
        # landing - when this loop is blocked waiting for the drone - are
        # recorded too.
        self.blocked = False
        self.latest = None
        self.done = threading.Event()
        display = threading.Thread(target=self._display_loop, daemon=True)
        display.start()
        airborne = landed = False
        try:
            flying = (getattr(drone, "is_flying", False) if self.airborne_at_start is None
                      else self.airborne_at_start)
            if not flying:
                self.state, self.blocked = "TAKEOFF", True
                drone.takeoff()
                self.state, self.blocked = "SEARCH", False
            airborne = True
            self.t_state = self.now()
            t_prev = self.now()
            while True:
                t = self.now()
                dt = min(max(t - t_prev, 0.01), 0.3)
                t_prev = t
                try:
                    st = drone.get_current_state() or {}
                    self.battery = st.get("bat", self.battery)
                    self.imu_yaw = st.get("yaw")      # helps pick the real pose over its mirror
                except Exception:
                    pass
                self.note = ""
                p = self.observe()
                if self.stop is not None and self.stop.is_set():
                    rc, action = self.finish("stopped by hand")
                else:
                    rc, action = self.step(p, self.battery, dt)
                self.rc = rc
                if action == "hop":
                    drone.send_rc_control(0, 0, 0, 0)
                    self.blocked = True
                    self.sleep(0.4)
                    try:
                        drone.go_xyz_speed(self.hop[0], self.hop[1], 0, HOP_SPEED)
                    except Exception as e:
                        self.say(f"hop refused ({e}) - landing here")
                    action = "land"
                if action == "land":
                    self.state, self.blocked = "LAND", True
                    drone.send_rc_control(0, 0, 0, 0)
                    drone.land()
                    airborne, landed = False, True
                    self.state = "LANDED"
                else:
                    drone.send_rc_control(*rc)
                self.write_log(t, p, rc, self.battery)
                if self.view and not self.window():
                    airborne = False
                    break
                if action == "land":
                    break
                self.sleep(max(0.0, 1.0 / RATE_HZ - (self.now() - t)))
            if landed:
                # keep the video going for a moment: the landing, and after it
                self.say(f"landed - recording {AFTER_LAND_S:.0f} s more")
                t_end = self.now() + AFTER_LAND_S
                while self.now() < t_end:
                    if self.view:
                        self.window()
                    self.sleep(0.05)
        finally:
            try:
                drone.send_rc_control(0, 0, 0, 0)
                if airborne:
                    drone.land()
            except Exception:
                pass
            self.done.set()
            display.join(timeout=2)
            if self.log:
                self.logf.close()
            if self.video:
                self.video.release()
                self.say(f"recording saved: {self.record}")
            if self.view:
                cv2.destroyAllWindows()
            if self.disconnect:
                try:
                    drone.streamoff()
                    drone.end()
                except Exception:
                    pass
        self.say(f"ended: {self.reason or 'loop exited'}")
        return self.reason

    def _display_loop(self):
        """Render, record and hand out the view at a steady RATE_HZ."""
        period, t_next = 1.0 / RATE_HZ, time.monotonic()
        while not self.done.is_set():
            try:
                img = self.render()
                self.latest = img
                if self.video:
                    self.video.write(img)
                if self.on_frame:
                    self.on_frame(img)
            except Exception as e:          # a drawing glitch must not stop the flight
                print(f"display: {e}")
            t_next += period
            time.sleep(max(0.0, t_next - time.monotonic()))

    def write_log(self, t, p, rc, battery):
        if not self.log:
            return

        def xyz(a):
            return ["", "", ""] if a is None else [f"{x:.1f}" for x in a]
        pose = self.pose if p is not None else None
        self.log.writerow([f"{t - self.t0:.3f}", self.target, self.state, *xyz(p),
                           *xyz(self.v if p is not None else None),
                           "" if pose is None else f"{pose.yaw_deg:.1f}",
                           "" if pose is None else int(pose.resolved),
                           *rc, "" if battery is None else battery, self.note])

    def render(self):
        """Camera with the overlay, next to the map: 960x720 + MAP_SIZE."""
        perc = self.perc[self.target]
        if self.blocked or self.frame is None:
            # the loop is waiting on the drone (takeoff, hop, land): show the
            # live picture without a stale pose on top
            frame, det, pose = self.frames.peek(), (None, None), None
        else:
            frame, det = self.frame, self.det
            pose = self.pose if perc.filt.fresh(self.now() - VIDEO_DELAY_S) else None
        if frame is None:
            frame = np.zeros((720, 960, 3), np.uint8)
        view = cv2.resize(perc.est.draw(frame, pose, *det), (960, 720))
        name = "board" if self.target == "board" else f"marker {self.target}"
        lines = [f"{self.state} ({name})   rc {self.rc}   battery {self.battery}%"]
        if self.err is not None and pose is not None and self.state not in ("SEARCH", "LAND"):
            e = self.err
            lines.append(f"error fwd {e['fwd']:+.0f} side {e['lat']:+.0f} up {e['up']:+.0f} cm"
                         f"  yaw {e['yaw']:+.0f} deg")
        for i, s in enumerate(reversed(lines)):
            y = 720 - 14 - 24 * i
            cv2.putText(view, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(view, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        p = perc.filt.position() if pose is not None else None
        yaw = pose.yaw_deg if pose is not None else 0.0
        side = perc.lmap.render(*MAP_SIZE, p, yaw, perc.trail, seen=pose is not None,
                                hud=perc.lmap.hud_lines(p, yaw, f"{self.state}  {name}"))
        return np.hstack([view, side])

    def window(self):
        """The --view window. False if the operator stopped the flight."""
        if self.latest is not None:
            cv2.imshow("landing - q lands, x cuts motors", self.latest)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) and self.state != "LANDED":
            self.say("stopped by hand - landing")
            self.reason = "stopped by hand"
            self.drone.send_rc_control(0, 0, 0, 0)
            self.drone.land()
            return False
        if k == ord("x"):
            self.say("EMERGENCY - motors off")
            self.reason = "emergency stop"
            self.drone.emergency()
            return False
        return True


def main():
    ap = argparse.ArgumentParser(description="Visit waypoint markers, then find the "
                                             "ArUco board and land on the pad.")
    ap.add_argument("--waypoints", type=int, nargs="*", default=list(WAYPOINTS), metavar="ID",
                    help="marker ids to visit first, in order (default: 4 5)")
    ap.add_argument("--marker-mm", type=float, default=150.0,
                    help="waypoint marker size, black edge to edge (default 150)")
    ap.add_argument("--view", action="store_true", help="show camera + map (q lands, x cuts motors)")
    ap.add_argument("--hold", action="store_true", help="stop at the near standoff, don't land")
    ap.add_argument("--log", metavar="FILE.csv", help="log every control step")
    ap.add_argument("--record", metavar="FILE.mp4",
                    help="save the camera + map view as a video (works without --view too)")
    ap.add_argument("--calib", default="tello_calib.npz")
    ap.add_argument("--allow-bad-calib", action="store_true",
                    help="fly even though the calibration fails its checks")
    args = ap.parse_args()

    board = load_board()
    print(f"target: {board.check()}")
    bad = [w for w in args.waypoints if w in board.ids]
    if bad:
        ap.error(f"ids {bad} belong to the landing board - waypoints must be other markers")
    if len(set(args.waypoints)) != len(args.waypoints):
        ap.error("each waypoint marker can be visited once")
    K, dist, trusted = load_calibration(args.calib)
    if not trusted and not args.allow_bad_calib:
        raise SystemExit("refusing to fly on this calibration - run calibrate_camera.py")
    Lander(open_drone(), K, dist, waypoints=args.waypoints, hold=args.hold,
           view=args.view, log=args.log, record=args.record, board=board,
           waypoint_marker_m=args.marker_mm / 1000).run()


if __name__ == "__main__":
    main()
