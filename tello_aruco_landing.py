"""
Visit waypoint markers, then search for the 4-marker ArUco board and land
on the pad in front of it.

    python tello_aruco_landing.py --view                    # markers 4 5 6 7, then the board, land
    python tello_aruco_landing.py --view --turns R R L R L  # which way to turn for each search
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
be visible from where the drone stops at the previous one - at the height
it stops at, too: it flies level with each marker's centre, and the camera
sees only about +-21 deg up and down.

The mission itself - states, standoffs, tolerances, timeouts, gains - is
landing_control.Mission / MissionConfig, the same code the GUI's Mission
tab runs. This file is the I/O around it: the drone, the video, the
localization of whichever target the mission wants, the log, the window.

Localization is tello_pose.py's, the same as the GUI: both mirror-image
poses of the target (PoseEstimator), the real one picked by BoardTracker
(image fit, gravity, IMU heading), smoothed by PoseFilter (Kalman). The
board is whatever board_config.py has configured (default: the 2x2 board).

Keys (with --view): q = land now, x = cut the motors (the drone DROPS),
a / d or the arrow keys = nudge the yaw left / right by hand (hold to keep
turning; the mission takes over again when you let go), n = skip the
current marker and search for the next target.

Safety: refuses to fly on a bad calibration, lands if the battery drops
below 15%, if the mission runs too long, if a marker isn't found within a
full turn, or if the hop it computes is implausible - and always lands on
exit. There is no obstacle sensing: keep the paths clear.
"""

import argparse
import csv
import dataclasses
import math
import threading
import time

import cv2
import numpy as np

from board_config import load_board
from landing_control import BOARD, Cmd, Mission, Obs, for_board, target_name
from tello_io import FrameGrabber, open_drone, put_text
from tello_map import LocalizationMap
from tello_pose import (BoardGeometry, BoardTracker, PoseEstimator, PoseFilter,
                        load_calibration)

# ----------------------------------------------------------------------
# The route
# ----------------------------------------------------------------------

# Waypoint markers in the order they are visited; the board comes last.
# While searching for each target the drone turns this way on the spot:
# +1 = right (clockwise), -1 = left. A full turn finds it either way - the
# right direction only makes it quicker. Set them for your room, here or
# with --turns.
WAYPOINTS = (4, 5, 6, 7)
SEARCH_TURN = {4: +1, 5: +1, 6: +1, 7: +1, BOARD: -1}
WAYPOINT_MARKER_M = 0.150   # print/room_marker_*_A4.pdf

# I/O timing. Everything about how the route is flown (standoffs, hover
# time, tolerances, timeouts, gains) is landing_control.MissionConfig.
RATE_HZ = 15.0
VIDEO_DELAY_S = 0.25        # the video lags; positions are predicted across it
# Yaw nudge by hand: each key press (or GUI button) overrides the mission's
# yaw command with this rate for NUDGE_S; holding the key keeps it going.
NUDGE_RC = 25
NUDGE_S = 0.4
AFTER_LAND_S = 5.0          # keep recording this long after touching down
MAP_SIZE = (564, 720)       # the map beside the 960x720 camera view

# One row per control step. The columns shared with the GUI's recordings
# (tello_gui.Recorder) have the same names, so tello_map.py replays both.
LOG_COLS = ["t", "target", "state", "px", "py", "pz", "vx", "vy", "vz",
            "yaw", "resolved", "source", "ambiguity", "rms", "n",
            "raw_x", "raw_y", "raw_z", "e_fwd", "e_lat", "e_up", "e_yaw",
            "a", "b", "c", "d", "hop_fwd", "hop_left",
            "bat", "h", "tof", "imu_pitch", "imu_roll", "imu_yaw", "vgx", "vgy", "vgz",
            "templ", "temph", "note"]

# The Tello lands by itself on a sagging battery or when it overheats.
# Height 0 and the downward sensor at its 10 cm floor this long: it is
# on the ground, so stop the mission instead of "searching" there.
GROUND_S = 1.0


def parse_turns(turns, waypoints):
    """'R R L R L' (or a list of them) -> {target: +1/-1}: one per waypoint
    marker in route order, then one for the board. Empty -> SEARCH_TURN,
    and right for any marker it does not list."""
    items = turns.replace(",", " ").split() if isinstance(turns, str) else list(turns or [])
    targets = [int(w) for w in waypoints] + [BOARD]
    if not items:
        return {t: SEARCH_TURN.get(t, 1) for t in targets}
    if len(items) != len(targets):
        raise ValueError(f"need {len(targets)} turns (one per waypoint marker, then the "
                         f"board), got {len(items)}")
    out = {}
    for t, s in zip(targets, items):
        s = str(s).strip().upper()
        if s not in ("R", "L", "RIGHT", "LEFT"):
            raise ValueError(f"turn {s!r}: use R or L")
        out[t] = 1 if s.startswith("R") else -1
    return out


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
    """Flies a landing_control.Mission on a real drone. The command line
    runs it with an OpenCV window (view=True); tello_gui.py runs it in a
    thread, reads each rendered view from `latest` (or gets it through
    on_frame) and each message through on_note, asks it to stop with the
    `stop` Event, and keeps the connection afterwards (disconnect=False).

    cfg: a MissionConfig (default: landing_control.for_board(board)).
    display: render the camera + map view on its own thread - default on
    when something shows or records it."""

    def __init__(self, drone, K, dist, waypoints=(), search_turn=None, hold=False, view=False,
                 log=None, record=None, on_frame=None, on_note=None, stop=None, disconnect=True,
                 airborne=None, board=None, waypoint_marker_m=WAYPOINT_MARKER_M, cfg=None,
                 display=None, now=time.monotonic, sleep=time.sleep):
        self.drone = drone
        board = board or load_board()
        cfg = cfg or for_board(board)[0]
        self.cfg = dataclasses.replace(cfg, hold=hold or cfg.hold)
        turns = parse_turns("", waypoints) if search_turn is None else search_turn
        self.mission = Mission(self.cfg, waypoints=waypoints, search_turn=turns)
        self.perc = {BOARD: Perception(K, dist, board, (("FAR", self.cfg.standoff_far),
                                                        ("NEAR", self.cfg.standoff_near)))}
        for w in waypoints:
            self.perc[int(w)] = Perception(K, dist, waypoint_geometry(board, w, waypoint_marker_m),
                                           (("STOP", self.cfg.waypoint_standoff),))
        # None: ask djitellopy. The GUI says, since it takes off with raw commands.
        self.airborne_at_start = airborne
        self.on_frame, self.on_note = on_frame, on_note
        self.stop, self.disconnect = stop, disconnect
        self.hold, self.view = hold, view
        self.display = (view or bool(record) or on_frame is not None) if display is None else display
        self.now, self.sleep = now, sleep
        self.frames = FrameGrabber(drone)
        self._target = None                  # whose track self.pose belongs to
        self.pose = None                     # latest pose of the target (orientation, resolved)
        self.raw = None                      # this step's pose, before filtering
        self.servo = None                    # this step's errors (landing_control.ServoOut)
        self.v = None
        self.det = (None, None)
        self.frame = None
        self.phase = None                    # TAKEOFF / HOP / LANDING / LANDED: waiting on the drone
        self.telem = {}
        self.battery, self.imu_yaw, self.rc = None, None, (0, 0, 0, 0)
        self.log = None
        if log:
            self.logf = open(log, "w", newline="")
            self.log = csv.writer(self.logf)
            self.log.writerow(LOG_COLS)
        self.record = record
        self.video = None                    # opened in run(): what the window shows, as .mp4
        self.reason = ""
        self.warned_upside_down = set()
        self.t0 = now()
        self.note = ""
        self.nudge = (0, 0.0)                # (direction, until): yaw override by hand
        self.skip_req = False                # skip the current marker, by hand
        self.nudging = False

    @property
    def target(self):
        return self.mission.target

    @property
    def state(self):
        return self.phase or self.mission.state.value

    def say(self, msg):
        line = f"[{self.now() - self.t0:6.1f}s] {msg}"
        print(line)
        self.note = msg
        if self.on_note:
            self.on_note(line)

    # -- one control step -------------------------------------------------
    def read_state(self):
        """The drone's state packet: battery, and the IMU yaw that helps
        pick the real pose over its mirror. Kept whole for the log."""
        try:
            st = self.drone.get_current_state() or {}
        except Exception:
            st = {}
        self.telem = st
        self.battery = st.get("bat", self.battery)
        self.imu_yaw = st.get("yaw")

    def on_ground(self, t):
        """True once the state packet has said "on the ground" (height 0,
        downward range at its 10 cm floor) for GROUND_S."""
        try:
            down = (float(self.telem.get("h", 99)) <= 0
                    and float(self.telem.get("tof", 999)) <= 10)
        except (TypeError, ValueError):
            down = False
        if not down:
            self.t_ground = None
            return False
        if self.t_ground is None:
            self.t_ground = t
        return t - self.t_ground >= GROUND_S

    def observe(self, now):
        """New frame -> pose of the mission's current target -> track.
        Returns the position predicted to now, or None if the target
        hasn't been seen recently."""
        target = self.mission.target
        if target != self._target:           # the route moved on: that track starts fresh
            self._target, self.pose = target, None
        perc = self.perc[target]
        frame, new = self.frames.grab()
        self.raw = None
        if new:
            self.frame = frame
            self.det = perc.est.detect(frame)[:2]
            t_cap = now - VIDEO_DELAY_S
            pose = perc.tracker.update(detection=self.det, t_capture=t_cap,
                                       imu_yaw=self.imu_yaw)
            self.raw = pose
            if pose is not None:
                self.pose = pose
                perc.filt.update(pose, now=t_cap)
                p = perc.filt.position()
                if p is not None:
                    perc.trail = (perc.trail + [tuple(p)])[-300:]
            elif (perc.est.upside_down(self.det)
                  and target not in self.warned_upside_down):
                self.warned_upside_down.add(target)
                what = "BOARD" if target == BOARD else f"MARKER {target}"
                self.say(f"!! THE {what} IS UPSIDE DOWN - mount it with the TOP mark up"
                         + (" (id 0 top left)" if target == BOARD else "")
                         + ". Ignoring it until then.")
        if self.pose is not None and perc.filt.fresh(now - VIDEO_DELAY_S):
            self.v = perc.filt.velocity()
            return perc.filt.predict(now)
        self.v = None
        return None

    # -- the loop ---------------------------------------------------------
    def run(self):
        drone, m = self.drone, self.mission
        self.t0 = self.now()
        self.imu_yaw, self.telem, self.t_ground = None, {}, None
        self.rc, self.battery = (0, 0, 0, 0), drone.get_battery()
        self.say(f"battery {self.battery}%   route: " + " -> ".join(
            f"{target_name(t)} ({'R' if m.search_turn.get(t, 1) > 0 else 'L'})"
            for t in m.targets))
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
        display = None
        if self.display:
            display = threading.Thread(target=self._display_loop, daemon=True)
            display.start()
        airborne = landed = False
        try:
            flying = (getattr(drone, "is_flying", False) if self.airborne_at_start is None
                      else self.airborne_at_start)
            if not flying:
                self.phase, self.blocked = "TAKEOFF", True
                drone.takeoff()
                self.phase, self.blocked = None, False
            airborne = True
            while True:
                t = self.now()
                self.read_state()
                self.note = ""
                p = self.observe(t)
                obs = Obs(t=t, p=p, v=self.v, pose=self.pose if p is not None else None,
                          battery=self.battery)
                if self.on_ground(t):
                    st = self.telem
                    m.abort(t, "the drone landed by itself")
                    self.say(f"!! THE DRONE LANDED BY ITSELF (battery {self.battery}%, "
                             f"temperature {st.get('templ', '?')}-{st.get('temph', '?')} C). "
                             "The Tello does this on a weak battery or when it overheats.")
                    self.write_log(t, p, Cmd(note="landed by itself"))
                    airborne, landed = False, True
                    self.phase = "LANDED"
                    break
                if self.stop is not None and self.stop.is_set():
                    cmd = m.abort(t, "stopped by hand")
                elif self.skip_req:
                    self.skip_req = False
                    cmd = m.skip(t)
                else:
                    cmd = m.step(obs)
                if cmd.note:
                    self.say(cmd.note)
                self.apply_nudge(t, cmd)
                self.servo, self.rc = cmd.servo, cmd.rc
                self.write_log(t, p, cmd)
                if cmd.action == "done":
                    break
                if cmd.action == "hop":
                    drone.send_rc_control(0, 0, 0, 0)
                    self.phase, self.blocked = "HOP", True
                    self.sleep(0.4)
                    try:
                        drone.go_xyz_speed(cmd.hop[0], cmd.hop[1], 0, self.cfg.hop_speed)
                    except Exception as e:
                        self.say(f"hop refused ({e}) - landing here")
                if cmd.action in ("hop", "land"):
                    self.phase, self.blocked = "LANDING", True
                    drone.send_rc_control(0, 0, 0, 0)
                    drone.land()
                    airborne, landed = False, True
                    self.phase = "LANDED"
                    break
                drone.send_rc_control(*cmd.rc)
                if self.view and not self.window():
                    airborne = False
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
            if display is not None:
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
        self.reason = self.reason or m.reason
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

    def write_log(self, t, p, cmd):
        if not self.log:
            return

        def num(v, fmt=".1f"):
            return "" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)

        def xyz(a):
            return ["", "", ""] if a is None else [f"{x:.1f}" for x in a]

        pose = self.pose if p is not None else None
        raw = self.raw
        s = cmd.servo
        st = self.telem
        self.log.writerow([
            f"{t - self.t0:.3f}", self.mission.target, self.state, *xyz(p),
            *xyz(self.v if p is not None else None),
            "" if pose is None else f"{pose.yaw_deg:.1f}",
            "" if pose is None else int(pose.resolved),
            "" if pose is None else pose.source,
            "" if pose is None else f"{pose.ambiguity:.2f}",
            "" if pose is None else f"{pose.reproj_rms_px:.2f}",
            "" if pose is None else pose.n_markers,
            *xyz(None if raw is None else raw.p_board_cm),
            *(["", "", "", ""] if s is None else
              [num(s.e_fwd), num(s.e_lat), num(s.e_up), num(s.e_yaw)]),
            *cmd.rc,
            *(["", ""] if cmd.hop is None else cmd.hop),
            *(st.get(k, "") for k in ("bat", "h", "tof", "pitch", "roll", "yaw",
                                      "vgx", "vgy", "vgz", "templ", "temph")),
            self.note])

    def render(self):
        """Camera with the overlay, next to the map: 960x720 + MAP_SIZE."""
        target = self.mission.target
        perc = self.perc[target]
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
        name = "board" if target == BOARD else f"marker {target}"
        lines = [f"{self.state} ({name})   rc {self.rc}   battery {self.battery}%"]
        s = self.servo
        if s is not None and pose is not None:
            side = "?" if math.isnan(s.e_lat) else f"{s.e_lat:+.0f}"
            lines.append(f"error fwd {s.e_fwd:+.0f} side {side} up {s.e_up:+.0f} cm"
                         f"  yaw {s.e_yaw:+.0f} deg")
        for i, line in enumerate(reversed(lines)):
            y = 720 - 14 - 24 * i
            put_text(view, line, (12, y), 0.6, (0, 255, 255))
        p = perc.filt.position() if pose is not None else None
        yaw = pose.yaw_deg if pose is not None else 0.0
        side = perc.lmap.render(*MAP_SIZE, p, yaw, perc.trail, seen=pose is not None,
                                hud=perc.lmap.hud_lines(p, yaw, f"{self.state}  {name}"))
        return np.hstack([view, side])

    def nudge_yaw(self, direction):
        """Turn left (-1) or right (+1) by hand for NUDGE_S, overriding the
        mission's yaw. Safe to call from another thread (the GUI)."""
        self.nudge = (1 if direction > 0 else -1, self.now() + NUDGE_S)

    def skip_target(self):
        """Skip the current waypoint marker at the next control step. Safe
        to call from another thread (the GUI)."""
        self.skip_req = True

    def apply_nudge(self, t, cmd):
        d, until = self.nudge
        active = bool(d) and t < until and cmd.action is None
        if active:
            cmd.rc = (*cmd.rc[:3], d * NUDGE_RC)
            if not self.nudging:
                self.say(f"yaw nudge {'right' if d > 0 else 'left'} (by hand)")
        self.nudging = active

    def window(self):
        """The --view window. False if the operator stopped the flight."""
        if self.latest is not None:
            cv2.imshow("landing - q lands, x cuts motors, a/d nudge yaw, n skips marker",
                       self.latest)
        kx = cv2.waitKeyEx(1)
        k = kx & 0xFF
        if k == ord("a") or kx in (2424832, 65361):      # a, left arrow
            self.nudge_yaw(-1)
        elif k == ord("d") or kx in (2555904, 65363):    # d, right arrow
            self.nudge_yaw(+1)
        elif k == ord("n"):
            self.skip_target()
        if k in (ord("q"), 27) and self.phase != "LANDED":
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
                    help="marker ids to visit first, in order (default: "
                         f"{' '.join(map(str, WAYPOINTS))})")
    ap.add_argument("--turns", nargs="*", default=[], metavar="R|L",
                    help="which way to turn when searching for each target: one per "
                         "waypoint marker, then one for the board (default: right for "
                         "the markers, left for the board)")
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
    try:
        turns = parse_turns(args.turns, args.waypoints)
    except ValueError as e:
        ap.error(f"--turns: {e}")
    K, dist, trusted = load_calibration(args.calib)
    if not trusted and not args.allow_bad_calib:
        raise SystemExit("refusing to fly on this calibration - run calibrate_camera.py")
    Lander(open_drone(), K, dist, waypoints=args.waypoints, search_turn=turns, hold=args.hold,
           view=args.view, log=args.log, record=args.record, board=board,
           waypoint_marker_m=args.marker_mm / 1000).run()


if __name__ == "__main__":
    main()
