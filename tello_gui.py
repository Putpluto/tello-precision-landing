"""
GUI for the ArUco precision-landing rig - one window for the whole bench.

Left: live video with the pose overlay. Right: a tab per job. Below:
telemetry, manual flight, and an AUTO HOLD mode that servos to a 3D
setpoint and stays there - click the Board tab's map to place it (top-down
panel: x and z, side panel: z and y) or type it in the x/y/z boxes.

    python tello_gui.py               # Tello, video + telemetry, flight enabled (= --live)
    python tello_gui.py --video f.mp4 # replay a recording through the pipeline

TABS
    Board      The localization map from tello_map.py - board-frame pose,
               top-down and side, the view that matters during an approach.
               Click/drag on it to move the AUTO HOLD setpoint; the dashed
               outline is where the board stays in the camera's view.
    Room map   Builds a RoomMap (map_room.py) from whatever the camera
               sees; Save writes --map. Fly MANUAL while it runs, or carry
               the drone by hand with the props off.
    Mission    The waypoint mission from tello_aruco_landing.py, run in
               this window: waypoint markers in order (100 cm in front,
               hover 2 s each), then the board, hop, land. Recorded to
               recordings/ including the landing and 5 s after it; the
               drone stays connected afterwards.
    Room loc   Localizes against that map (room_pose.py), opens the 3D
               view.
    Calibrate  Chessboard capture and solve (calibrate_camera.py), so
               intrinsics never need a separate CLI round trip.

Flight mode (IDLE/MANUAL/HOLD) is independent of the tab: building a room
map while flying manually is the normal way to do it.

SCOPE
    Monitoring, manual flight, and the hold-at-standoff servo used for
    tuning - the same thing tello_aruco_landing.py --hold does, reusing
    landing_control's ServoLaw and gains and tello_pose's BoardTracker and
    PoseFilter rather than a second copy of any of them.
    The mission is not reimplemented here: the Mission tab runs
    tello_aruco_landing.Lander - the same object the command line flies -
    in a thread, on this window's connection.

SAFETY
    Every send goes through one lock, because djitellopy has none and pops
    replies off a shared deque - two senders steal each other's answers.
    Closing the window stops the motors and lands. AUTO HOLD refuses to arm
    on nominal intrinsics, for the same reason tello_aruco_landing.py does:
    a wrongly-scaled pose looks perfectly confident.
"""

import argparse
import glob
import math
import pathlib
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

import calibrate_camera as calib
from board_config import load_board
from landing_control import ServoLaw, clamp_setpoint, default_gains, for_board
from map_room import RoomMapper
from room_map import RoomMap
from room_pose import RoomLocalizer
from room_view import RoomView3D
from tello_io import file_frames, open_drone, tello_frames
from tello_map import LocalizationMap
from tello_pose import BoardTracker, PoseEstimator, PoseFilter, load_calibration
from tello_aruco_landing import Lander

try:
    from PIL import Image, ImageTk
    _HAVE_PIL = True
except ImportError:                     # PPM fallback, Tk decodes it natively
    _HAVE_PIL = False

VIDEO_W, VIDEO_H = 640, 480
MAP_W, MAP_H = 470, 600
VIEW3D_W, VIEW3D_H = 470, 470
TRAIL_LEN = 300
UI_HZ = 20
HOLD_LATENCY_S = 0.25          # video delay AUTO HOLD predicts across


# ----------------------------------------------------------------------
# Image plumbing
# ----------------------------------------------------------------------

def to_photo(bgr):
    """numpy BGR -> a Tk image. PIL if present, raw PPM otherwise."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if _HAVE_PIL:
        return ImageTk.PhotoImage(Image.fromarray(rgb))
    h, w = rgb.shape[:2]
    return tk.PhotoImage(data=b"P6 %d %d 255 " % (w, h) + rgb.tobytes())


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------

class Recorder:
    """What the window shows, to an MP4, plus one CSV row per video frame.

    The video runs at a fixed rate in wall-clock time: each write appends
    as many copies of the current picture as the clock calls for, so it
    plays back at real speed however fast the UI managed to redraw. The
    CSV uses tello_aruco_landing.py --log's column names where they
    overlap, and row i belongs to frame i - so a camera-only recording
    replays in tello_map.py exactly like a landing run:

        python tello_map.py --log recordings/X.csv --video recordings/X.mp4
    """

    CAM = (960, 720)               # the camera view is recorded full size
    PANEL = (MAP_W, MAP_H)         # the side panel, padded to one size
    COLS = ["t", "state", "tab", "px", "py", "pz", "yaw", "rms", "n", "source",
            "a", "b", "c", "d", "spx", "spy", "spz", "bat", "h", "tof",
            "imu_yaw", "rx", "ry", "rz"]

    def __init__(self, folder="recordings", fps=20.0, with_panel=True, stem=None):
        import csv
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.with_panel = with_panel
        stem = stem or time.strftime("%Y-%m-%d_%H-%M-%S") + "_gui"
        self.video_path = self.folder / f"{stem}.mp4"
        self.csv_path = self.folder / f"{stem}.csv"
        self._csvf = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csvf)
        self._csv.writerow(self.COLS)
        self._vw = None
        self.size = None
        self.t0 = None
        self.frames = 0

    def compose(self, view, panel=None):
        """One recorded picture: the camera view, and beside it (if
        with_panel) the side panel, both at the panel's height."""
        cam = view if (view.shape[1], view.shape[0]) == self.CAM else cv2.resize(view, self.CAM)
        if not self.with_panel:
            return cam
        pw, ph = self.PANEL
        right = np.full((ph, pw, 3), 26, np.uint8)
        if panel is not None:
            h, w = min(panel.shape[0], ph), min(panel.shape[1], pw)
            right[:h, :w] = panel[:h, :w]
        cw = int(round(cam.shape[1] * ph / cam.shape[0])) // 2 * 2
        return np.hstack([cv2.resize(cam, (cw, ph)), right])

    def write(self, view, panel=None, row=(), now=None):
        """Bring the recording up to wall-clock time `now`. Frame slots the
        UI missed (it stalled, or ticks slower than fps) repeat what was on
        screen during them - the previous picture - and this one takes the
        slot it appeared in."""
        now = time.time() if now is None else now
        img = self.compose(view, panel)
        if self._vw is None:
            self.size = (img.shape[1], img.shape[0])
            self._vw = cv2.VideoWriter(str(self.video_path),
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       self.fps, self.size)
            if not self._vw.isOpened():
                raise RuntimeError(f"cannot write {self.video_path}")
            self.t0 = now
            self._prev = (img, list(row))
        # (+1e-6: a tick landing exactly on a frame boundary counts for it)
        due = int((now - self.t0) * self.fps + 1e-6) + 1 - self.frames
        due = min(due, int(self.fps))                   # catch up <= 1 s
        for k in range(max(0, due)):
            pic, r = self._prev if k < due - 1 else (img, list(row))
            self._vw.write(pic)
            self._csv.writerow([f"{self.frames / self.fps:.3f}", *r])
            self.frames += 1
        self._prev = (img, list(row))

    @property
    def seconds(self):
        return self.frames / self.fps

    def close(self):
        if self._vw is not None:
            self._vw.release()
        self._csvf.close()
        return (f"{self.video_path} + {self.csv_path.name}  "
                f"({self.frames} frames, {self.seconds:.1f} s)")


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

class App:
    def __init__(self, args):
        self.args = args
        self.live = args.live                 # a drone to fly
        self.geom = load_board()              # board.json (board_config.py) or the 2x2 board
        cfg, self.sp_limits = for_board(self.geom)
        self.standoff_far = args.standoff_far or cfg.standoff_far
        self.standoff_near = args.standoff_near or cfg.standoff_near
        self.K, self.dist, self.real_calib = load_calibration(args.calib)
        self.cx = float(self.K[0, 2])
        self.est = PoseEstimator(self.K, self.dist, self.geom)
        self.tracker = BoardTracker(self.est)
        self.pf = PoseFilter(target_span_m=self.geom.span_m)
        self.lmap = LocalizationMap(self.geom,
                                    standoffs=(("FAR", self.standoff_far),
                                               ("NEAR", self.standoff_near)))

        self.lock = threading.Lock()        # serialises every drone send
        self.state_lock = threading.Lock()  # guards the fields below
        self.stop = threading.Event()
        self.msgs = queue.Queue()

        self.frame = None
        self.det = (None, None)
        self.n_frames = 0
        self.pose = None
        self.p_filt = None
        self.trail = deque(maxlen=TRAIL_LEN)
        self.fps = 0.0
        self.telem = {}
        self.rc = (0, 0, 0, 0)
        self.manual = [0, 0, 0, 0]
        self.mode = "IDLE"                  # IDLE | MANUAL | HOLD
        self.airborne = False
        self.speed = 30
        # AUTO HOLD target, (x, y, z) cm in the board's level frame. Set it
        # by clicking the Board tab's map (top-down: x and z; side: z and
        # y) or in the x/y/z boxes; it is always kept where the board stays
        # in view (landing_control.SetpointLimits).
        self.setpoint = self._clamp_sp((0.0, 0.0, self.standoff_near))[0]
        # Board tab shows the 2D map (click to place the setpoint) or a 3D
        # view of the same thing (drag to orbit) - whichever is up is what
        # a recording captures.
        self.board_view = "2d"
        self.board3d = RoomView3D(self.geom)
        self.board_scene = RoomMap({}, board_R=self.geom.R_level, board_t=np.zeros(3))
        self._drag3d = None
        self.frame_board3d()
        self._sp_notes = []
        self.tello = None
        self.ctl = None
        self.mission = None                 # (Lander, stop Event) while one flies
        self._last_send = 0.0
        self._photo = [None, None]          # keep refs or Tk garbage-collects

        # -- tabs: which job the worker does with each frame --------------
        self.task = "board"                 # board | roommap | roomloc | calib
        self.mapper = self._new_mapper()
        self.room_map = None
        self.localizer = None
        self.room_pose = None
        self._load_room_map(quiet=True)

        # 3D view. While mapping, the map is re-solved from the graph every
        # so often so the view fills in as markers connect, instead of
        # showing nothing until Save.
        self.view3d = RoomView3D(self.geom)
        self.draft_map = None
        self.draft_loc = None
        self.room_trail = deque(maxlen=TRAIL_LEN)
        self._draft_every = max(1, int(UI_HZ))      # ~1 s
        self._draft_tick = 0
        # build() re-picks the most-seen marker as root each time, and the
        # root is the origin - so letting it drift re-origins the whole map
        # mid-session and the view jumps. Pin the first one we get, and
        # save with it too, so the file matches what you were looking at.
        self._draft_root = None

        self.calib_dir = pathlib.Path(args.calib_dir)
        self.calib_corners = None           # live chessboard preview
        self.calib_cells = set()            # image thirds the board has covered
        self.calib_shots = len(glob.glob(str(self.calib_dir / "calib_*.png")))
        self.calib_result = None
        self._calib_skip = 0

        # -- video panel options (right-click menu) ----------------------
        # Plain attributes, not Tk vars: compose() runs headless under
        # --selftest, where no Tk root exists.
        self.overlay = "full"               # full | markers | clean
        self.show_hud = True
        self.downvision = False

        # -- recording (the Rec button) -----------------------------------
        self.rec = None                     # a Recorder while recording
        self.rec_with_panel = True          # camera + side panel, or camera only
        self._full_view = None              # last composed camera view, full size
        self._right = None                  # last side-panel image

    # -- logging ------------------------------------------------------
    def log(self, s):
        self.msgs.put(f"{time.strftime('%H:%M:%S')}  {s}")

    # -- drone --------------------------------------------------------
    def _send(self, cmd, expect_reply=True, timeout=7):
        if self.tello is None:
            return ""
        with self.lock:
            self._last_send = time.time()
            if not expect_reply:
                self.tello.send_command_without_return(cmd)
                return ""
            return self.tello.send_command_with_return(cmd, timeout=timeout)

    def _rc(self, a, b, c, d):
        a, b, c, d = (int(np.clip(v, -100, 100)) for v in (a, b, c, d))
        self._send(f"rc {a} {b} {c} {d}", expect_reply=False)
        with self.state_lock:
            self.rc = (a, b, c, d)

    def connect(self):
        self.tello = open_drone()
        self.log(f"connected, battery {self.tello.get_battery()}%")
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _keepalive(self):
        """The drone auto-lands after 15 s of silence. rc traffic already
        feeds that watchdog, so only ping when we have been idle."""
        mode = "keepalive"
        while not self.stop.wait(3.0):
            if time.time() - self._last_send < 8.0 or self.mission is not None:
                continue                    # a mission keeps the link busy itself
            try:
                if mode == "keepalive":
                    if "ok" not in str(self._send("keepalive")).lower():
                        mode = "battery?"   # SDK 2.0 only; older firmware 400s
                        self.log("keepalive unsupported, using battery?")
                else:
                    self._send("battery?")
            except Exception as e:
                self.log(f"keepalive failed: {e}")

    # -- frame sources ------------------------------------------------
    def _frames(self):
        if self.live:
            # BGR, placeholder skipped, each frame once (tello_io docstring)
            yield from tello_frames(self.tello, stop=self.stop)
        elif self.args.video:
            for f in file_frames(self.args.video, loop=True, stop=self.stop):
                yield f
                time.sleep(1.0 / 30.0)

    # -- worker -------------------------------------------------------
    def _worker(self):
        last = time.time()
        for frame in self._frames():
            if self.stop.is_set():
                break
            if frame is None:
                time.sleep(0.02)
                continue
            if self.mission is not None:
                # the mission does its own localization on these frames
                try:
                    self.telem = self.tello.get_current_state() or {}
                except Exception:
                    pass
                continue
            # Detect once here and hand the result to every estimator and
            # the overlay - re-detecting on the Tk thread stalls the UI,
            # and at long-range settings detectMarkers dominates the loop.
            corners, ids, _ = self.est.detect(frame)
            imu_yaw = self.telem.get("yaw") if self.live else None
            now = time.time()
            t_cap = now - (HOLD_LATENCY_S if self.live else 0.0)
            pose = self.tracker.update(detection=(corners, ids), t_capture=t_cap,
                                       imu_yaw=imu_yaw)
            p = self.pf.update(pose, now=t_cap)

            with self.state_lock:
                task = self.task
            room_pose = None
            if task == "roommap":
                self.mapper.add_frame(frame, detection=(corners, ids))
                room_pose = self._draft_step((corners, ids), frame)
            elif task == "roomloc" and self.localizer is not None:
                room_pose = self.localizer.estimate(frame,
                                                    detection=(corners, ids))
            elif task == "calib":
                self._calib_step(frame)
            if room_pose is not None:
                self.room_trail.append(tuple(float(v)
                                             for v in room_pose.p_world_cm))

            now = time.time()
            dt = now - last
            last = now

            with self.state_lock:
                self.frame = frame
                self.det = (corners, ids)
                self.n_frames += 1
                self.pose = pose
                self.room_pose = room_pose
                self.p_filt = None if p is None else np.asarray(p, float)
                if pose is not None and p is not None:
                    self.trail.append(tuple(float(v) for v in p))
                self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(dt, 1e-3))
                mode = self.mode

            if self.live:
                try:
                    self.telem = self.tello.get_current_state() or {}
                except Exception:
                    pass
                if mode == "MANUAL":
                    self._rc(*self.manual)
                elif mode == "HOLD":
                    self._hold_step(pose, p)

    # -- room map ------------------------------------------------------
    def _new_mapper(self):
        """A RoomMapper for the configured target. A single-marker target
        is the board on its own, so one marker of it is enough to place it."""
        return RoomMapper(self.K, self.dist, self.args.marker_size, geom=self.geom,
                          dict_id=self.geom.dict_id,
                          min_board_markers=1 if self.geom.single else 2)

    def _new_localizer(self, room_map):
        return RoomLocalizer(self.K, self.dist, room_map, geom=self.geom,
                             dict_id=self.geom.dict_id)

    def _load_room_map(self, quiet=False):
        try:
            self.room_map = RoomMap.load(self.args.map)
            self.localizer = self._new_localizer(self.room_map)
            if not quiet:
                self.log(f"loaded {self.room_map}")
            return True
        except Exception as e:
            self.localizer = None
            if not quiet:
                self.log(f"no map at {self.args.map}: {e}")
            return False

    def cmd_save_map(self):
        try:
            room_map, root_id = self.mapper.build(root_id=self._draft_root)
        except Exception as e:
            self.log(f"save failed: {e}")
            return
        room_map.save(self.args.map)
        self.log(f"wrote {self.args.map} ({len(room_map)} markers, "
                 f"root {root_id}, "
                 f"{'board included' if room_map.has_board else 'NO BOARD'})")
        if not room_map.has_board:
            self.log("  without the board there is no coarse approach - "
                     "re-map with the board and a room marker in one frame")
        self._load_room_map(quiet=True)

    def cmd_reset_map(self):
        self.mapper = self._new_mapper()
        self.draft_map = self.draft_loc = None
        self._draft_root = None
        self.room_trail.clear()
        self.view3d._framed_n = -1
        self.view3d._user_moved = False
        self.log("room mapping reset")

    def _draft_step(self, detection, frame):
        """Re-solve the half-built map now and then, and locate the drone
        in it, so the 3D view shows the room taking shape while you walk
        rather than nothing until Save. Cheap: the graph is tiny and the
        detection is already paid for."""
        self._draft_tick += 1
        if self._draft_tick % self._draft_every == 0:
            try:
                draft, root = self.mapper.build(root_id=self._draft_root,
                                                verbose=False)
                self._draft_root = root
            except Exception:
                draft = None
            if draft is not None and len(draft):
                self.draft_map = draft
                if self.draft_loc is None:
                    self.draft_loc = self._new_localizer(draft)
                else:
                    self.draft_loc.set_map(draft)
                self.view3d.frame_map(draft)
        if self.draft_loc is None:
            return None
        return self.draft_loc.estimate(frame, detection=detection)

    def cmd_open_3d(self):
        if self.localizer is None:
            self.log("load or save a map first")
            return
        cmd = [sys.executable, "view_room_3d.py", "--map", self.args.map,
               "--calib", self.args.calib]
        if self.live:
            cmd += ["--live"]
        elif self.args.video:
            cmd += ["--video", self.args.video]
        else:
            self.log("3D view needs --live or --video")
            return
        subprocess.Popen(cmd)
        self.log("opened view_room_3d.py")

    # -- mission ----------------------------------------------------------
    def cmd_mission_start(self):
        """Fly tello_aruco_landing's mission from this window: the waypoint
        markers in order (100 cm in front of each, hover 2 s), then the
        board, hop, land. Same code as the command line, on this window's
        connection - which it keeps afterwards."""
        if not self.live or self.tello is None:
            self.log("the mission needs the drone (--live)")
            return
        if self.mission is not None:
            return
        if not self.real_calib and not messagebox.askyesno(
                "Calibration", "tello_calib.npz is missing or fails its checks - "
                "distances may be wrong. Fly anyway?"):
            return
        try:
            wps = [int(v) for v in self.wp_var.get().replace(",", " ").split()]
        except ValueError:
            messagebox.showerror("Mission", "Waypoints are marker numbers, e.g. 6 5")
            return
        if [w for w in wps if w in self.geom.ids] or len(set(wps)) != len(wps):
            messagebox.showerror("Mission", "Waypoints must be different markers, "
                                            f"not the board's ids {list(self.geom.ids)}")
            return
        hold = bool(self.mission_hold_var.get())
        route = " -> ".join([f"marker {w}" for w in wps] + ["board"])
        ending = "HOLD 70 cm in front of the board (no hop, no landing)" if hold \
            else "hop over the pad and LAND"
        if not messagebox.askyesno(
                "Start mission?",
                f"This flies the drone autonomously:\n\n"
                f"{'' if self.airborne else 'take off, '}{route}, then {ending}.\n\n"
                "Stop mission (or ESC / L) lands at once; x cuts the motors.\n\nProceed?"):
            return
        self.set_mode("IDLE")
        rec = log = None
        if self.mission_rec_var.get():
            pathlib.Path("recordings").mkdir(exist_ok=True)
            stem = str(pathlib.Path("recordings") /
                       (time.strftime("%Y-%m-%d_%H-%M-%S") + "_mission"))
            rec, log = stem + ".mp4", stem + ".csv"
        stop = threading.Event()
        lander = Lander(self.tello, self.K, self.dist, waypoints=wps, hold=hold,
                        log=log, record=rec, on_note=self.log, stop=stop,
                        disconnect=False, airborne=self.airborne, board=self.geom,
                        waypoint_marker_m=self.args.marker_size)
        self.mission = (lander, stop)
        threading.Thread(target=self._run_mission, args=(lander,), daemon=True).start()

    def _run_mission(self, lander):
        try:
            with self.lock:                 # nothing else sends while it flies
                lander.run()
        except Exception as e:
            self.log(f"mission error: {e} - landing")
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.land()
            except Exception:
                pass
        finally:
            self.airborne = False           # it always ends on the ground
            self.tracker.reset()
            self.pf.reset()
            self._last_send = time.time()
            self.mission = None

    def cmd_mission_stop(self):
        if self.mission is not None:
            self.mission[1].set()
            self.log("stop requested - landing")

    # -- calibration ----------------------------------------------------
    def _calib_step(self, frame):
        """Live chessboard preview. Throttled: findChessboardCorners on a
        960x720 frame is far slower than the ArUco pass, and a preview
        only has to keep up with the eye."""
        self._calib_skip = (self._calib_skip + 1) % 3
        if self._calib_skip:
            return
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.calib_corners = calib.find_corners(gray, refine=False, fast=True)

    def _calib_cell(self, corners, shape):
        """Which third-of-the-frame the board's centre landed in. Coverage
        of the corners is what fixes distortion; frames all dead centre
        give a good focal length and a bad lens model."""
        c = corners.reshape(-1, 2).mean(axis=0)
        col = min(2, int(c[0] / (shape[1] / 3)))
        row = min(2, int(c[1] / (shape[0] / 3)))
        return row * 3 + col

    def cmd_calib_snap(self):
        with self.state_lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = calib.find_corners(gray, refine=False)
        if corners is None:
            self.log("no chessboard in this frame - not saved")
            return
        self.calib_dir.mkdir(exist_ok=True)
        fn = self.calib_dir / f"calib_{self.calib_shots:03d}.png"
        cv2.imwrite(str(fn), frame)
        self.calib_shots += 1
        self.calib_cells.add(self._calib_cell(corners, frame.shape))
        self.log(f"saved {fn}  ({self.calib_shots} shots, "
                 f"{len(self.calib_cells)}/9 regions covered)")

    def cmd_calib_clear(self):
        self.calib_shots = 0
        self.calib_cells.clear()
        self.calib_result = None
        self.log(f"counter reset - existing files in {self.calib_dir} kept")

    def cmd_calib_solve(self):
        files = sorted(glob.glob(str(self.calib_dir / "calib_*.png")))
        if not files:
            self.log(f"no frames in {self.calib_dir}")
            return
        self.log(f"solving on {len(files)} frames...")
        try:
            r = calib.calibrate(files)
        except Exception as e:
            self.log(f"calibration failed: {e}")
            return
        fx, fy = calib.fov_deg(r["K"], r["size"])
        self.calib_result = dict(rms=r["rms"], K=r["K"], dist=r["dist"],
                                 size=r["size"], n=len(r["used"]),
                                 total=len(files), fov=(fx, fy),
                                 worst=max(r["errs"]), std=r["std"],
                                 problems=r["problems"])
        calib.save_calibration(self.args.calib, r)
        self.log(f"wrote {self.args.calib}  rms {r['rms']:.3f} px  "
                 f"fov {fx:.1f}x{fy:.1f} deg")
        # the view-diversity checks count too, not just the numbers
        self._apply_calibration(r["K"], r["dist"], r["problems"])

    def _apply_calibration(self, K, dist, problems=None):
        """Swap the new intrinsics into everything that holds a copy, so
        the rest of the window starts using them without a restart."""
        from tello_pose import calibration_problems
        self.K, self.dist = np.asarray(K), np.asarray(dist)
        if problems is None:
            problems = calibration_problems(self.K, self.dist)
        self.real_calib = not problems
        self.cx = float(self.K[0, 2])
        self.est = PoseEstimator(self.K, self.dist, self.geom)
        self.tracker = BoardTracker(self.est)
        self.pf.reset()
        self.mapper = self._new_mapper()
        self._load_room_map(quiet=True)
        if problems:
            self.log("intrinsics applied, but they fail the sanity checks - "
                     "AUTO HOLD stays disabled:")
            for p in problems:
                self.log(f"   {p}")
        else:
            self.log("intrinsics applied - room mapping reset, AUTO HOLD enabled")

    def _hold_step(self, pose, p):
        """Servo to the setpoint and stay - the same law, tracker and
        filter tello_aruco_landing.py flies with. The nose stays on the
        board, so an off-axis setpoint is held facing it."""
        if p is None or pose is None or self.ctl is None:
            self._rc(0, 0, 0, 0)          # hover through a dropout
            return
        now = time.time()
        dt = min(max(now - getattr(self, "_t_hold", now - 1 / 15), 0.01), 0.3)
        self._t_hold = now
        p_now = self.pf.predict(now)
        target = np.array(self.setpoint, float)   # replaced whole by the UI: atomic
        out = self.ctl.step(p_now, self.pf.velocity(), pose.R_level_cam,
                            target, np.zeros(3), lateral=pose.resolved, dt=dt)
        if out is None:
            self._rc(0, 0, 0, 0)
            return
        self._rc(*out.rc)

    # -- setpoint ------------------------------------------------------
    def _clamp_sp(self, sp):
        return clamp_setpoint(sp, self.sp_limits, self.geom.pad_drop_m * 100.0)

    def set_setpoint(self, sp, announce=True):
        """Clamp to where the board stays in view, store, show. The worker
        reads self.setpoint from another thread; replacing the whole array
        (never editing it in place) keeps that safe."""
        sp, notes = self._clamp_sp(sp)
        self.setpoint = sp
        self._sp_notes = notes
        if getattr(self, "sp_vars", None):
            for v, val in zip(self.sp_vars, sp):
                v.set(f"{val:.0f}")
        if announce:
            self.log(f"setpoint x {sp[0]:+.0f} y {sp[1]:+.0f} z {sp[2]:.0f} cm"
                     + ("" if self.mode == "HOLD" else "  (AUTO HOLD flies to it)"))
            for n in notes:
                self.log(f"   {n}")

    def _on_sp_entry(self, _=None):
        try:
            sp = [float(v.get()) for v in self.sp_vars]
        except ValueError:
            self.set_setpoint(self.setpoint, announce=False)   # put the numbers back
            return
        if np.allclose(sp, self.setpoint, atol=0.5):
            return
        self.set_setpoint(sp)

    def cmd_sp_reset(self):
        self.set_setpoint((0.0, 0.0, self.standoff_near))

    def _map_xy(self, ev):
        """Label pixel -> map image pixel. The image is centred in the
        label, which may be a few pixels bigger (border, padding)."""
        img_w, img_h = MAP_W, MAP_H
        dx = (self.map_lbl.winfo_width() - img_w) / 2.0
        dy = (self.map_lbl.winfo_height() - img_h) / 2.0
        return ev.x - dx, ev.y - dy

    # -- board 3D view ----------------------------------------------------
    def frame_board3d(self):
        """Board, pad and the approach out to FAR, seen from front-right."""
        g = self.geom
        far = self.standoff_far / 100.0
        self.board3d.az, self.board3d.el = math.radians(40.0), math.radians(28.0)
        self.board3d.frame_points([[0, 0, 0], [0, -g.pad_drop_m, g.pad_out_m],
                                   [-0.4, 0.3, far], [0.4, 0.3, far]])
        self.board3d._user_moved = False

    def render_board3d(self):
        with self.state_lock:
            pose = self.pose
            p = None if self.p_filt is None else self.p_filt.copy()
            trail = list(self.trail)
            mode = self.mode
        yaw = None if pose is None else pose.yaw_deg
        hud = self.lmap.hud_lines(p, yaw, f"{mode}   3D: drag to orbit, wheel to zoom",
                                  setpoint=self.setpoint)
        return self.board3d.render(
            MAP_W, MAP_H, self.board_scene, p_world_cm=p,
            R_world_cam=None if pose is None else pose.R_level_cam,
            trail=trail, seen=pose is not None, hud=hud, setpoint_cm=self.setpoint)

    def _set_board_view(self):
        self.board_view = self.board_view_var.get()
        self.log("board tab: 3D view - drag to orbit, wheel to zoom (setpoint: "
                 "use the x/y/z boxes, or switch back to 2D to click it)"
                 if self.board_view == "3d" else "board tab: 2D map - click to set the setpoint")

    def _on_map_wheel(self, ev):
        if self.board_view == "3d":
            self.board3d.zoom(0.9 if ev.delta > 0 else 1.1)

    def _on_map_click(self, ev, release=False):
        """2D: top-down panel sets x and z; side panel sets z and y. The
        other coordinate is kept, so two clicks place any point in 3D.
        3D: drag orbits the view."""
        if self.board_view == "3d":
            if release:
                self._drag3d = None
                return
            if self._drag3d is not None:
                x0, y0 = self._drag3d
                self.board3d.orbit(-(ev.x - x0) * 0.6, (ev.y - y0) * 0.6)
            self._drag3d = (ev.x, ev.y)
            return
        hit = self.lmap.to_world(*self._map_xy(ev))
        if hit is None:
            return
        panel, a, b = hit
        x, y, z = self.setpoint
        if panel == "top":
            x, z = a, b
        else:
            z, y = a, b
        # a drag updates silently; the release says where it ended up
        self.set_setpoint((x, y, z), announce=release)

    # -- commands -----------------------------------------------------
    def cmd_takeoff(self):
        if self.mission is not None:
            return
        self.log("takeoff")
        threading.Thread(target=self._takeoff, daemon=True).start()

    def _takeoff(self):
        try:
            r = self._send("takeoff", timeout=20)
            self.log(f"takeoff -> {r}")
            if "ok" in str(r).lower():
                self.airborne = True
        except Exception as e:
            self.log(f"takeoff failed: {e}")

    def cmd_land(self):
        if self.mission is not None:
            self.cmd_mission_stop()         # the mission lands
            return
        self.set_mode("IDLE")
        self.manual = [0, 0, 0, 0]
        self._rc(0, 0, 0, 0)
        self.log("land")
        threading.Thread(target=self._land, daemon=True).start()

    def _land(self):
        try:
            self.log(f"land -> {self._send('land', timeout=20)}")
        except Exception as e:
            self.log(f"land failed: {e}")
        self.airborne = False

    def cmd_emergency(self):
        # Cuts the motors instantly: the drone DROPS. Only worth it when the
        # alternative is hitting something.
        self.log("EMERGENCY - motors off")
        if self.mission is not None:
            self.mission[1].set()
        try:
            # not through _send: a mission holds that lock, and this is
            # fire-and-forget anyway
            self.tello.emergency()
        except Exception as e:
            self.log(f"emergency failed: {e}")
        if self.mission is None:
            self.set_mode("IDLE")
        self.airborne = False

    def cmd_hover(self):
        if self.mission is not None:
            return
        self.manual = [0, 0, 0, 0]
        self._rc(0, 0, 0, 0)

    def set_mode(self, mode):
        if self.mission is not None:
            self.mode_var.set("IDLE")
            self.log("a mission is flying - Stop mission first")
            return
        if mode == "HOLD" and not self.real_calib:
            self.log("HOLD refused: intrinsics are nominal or fail the sanity "
                     "checks - run calibrate_camera.py first")
            self.mode_var.set(self.mode)
            return
        if mode in ("HOLD", "MANUAL") and not self.live:
            self.mode_var.set("IDLE")
            return
        if mode == "HOLD":
            self.ctl = ServoLaw(default_gains())
        with self.state_lock:
            self.mode = mode
        if mode == "IDLE":
            self.manual = [0, 0, 0, 0]
            self._rc(0, 0, 0, 0)
        self.mode_var.set(mode)
        self.log(f"mode -> {mode}")

    # -- keyboard -----------------------------------------------------
    def on_key(self, ev):
        k = ev.keysym.lower()
        if k == "escape":
            self.cmd_land()
            return
        if not self.live:
            return
        if self.mission is not None:
            if k == "l":
                self.cmd_land()             # = stop the mission, it lands
            elif k == "x":
                self.cmd_emergency()
            return
        # The binding is on the root window, so it also sees keys typed into
        # the setpoint boxes and the --log field. Typing "run.csv" there
        # must not fly the drone ('r' climbs, 's' backs off).
        if isinstance(ev.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Spinbox)):
            return
        if k in ("w", "s", "a", "d", "r", "f", "q", "e", "space"):
            if self.mode != "MANUAL":
                self.set_mode("MANUAL")
            v = self.speed
            m = self.manual
            # Sticky, exactly like tello_cli.py --keys: key-repeat is not
            # reliably detectable, so a key sets the velocity until changed.
            if k == "space":
                self.manual = [0, 0, 0, 0]
            elif k == "w":
                m[1] = v
            elif k == "s":
                m[1] = -v
            elif k == "a":
                m[0] = -v
            elif k == "d":
                m[0] = v
            elif k == "r":
                m[2] = v
            elif k == "f":
                m[2] = -v
            elif k == "q":
                m[3] = -v
            elif k == "e":
                m[3] = v
        elif k == "t":
            self.cmd_takeoff()
        elif k == "l":
            self.cmd_land()
        elif k == "x":
            self.cmd_emergency()

    # -- rendering ----------------------------------------------------
    def compose(self):
        """Returns (video_bgr, map_bgr) sized for the panels."""
        with self.state_lock:
            frame = None if self.frame is None else self.frame.copy()
            corners, ids = self.det
            pose = self.pose
            p = None if self.p_filt is None else self.p_filt.copy()
            trail = list(self.trail)
            fps, mode, rc = self.fps, self.mode, self.rc
            task = self.task

        if frame is None:
            view = np.full((VIDEO_H, VIDEO_W, 3), 30, np.uint8)
            cv2.putText(view, "waiting for video", (150, VIDEO_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            self._full_view = view
        else:
            if self.overlay == "clean":
                view = frame.copy()          # plain camera, nothing drawn
            elif self.overlay == "markers":
                view = frame.copy()
                if corners is not None and ids is not None:
                    cv2.aruco.drawDetectedMarkers(view, corners, ids)
            elif task == "calib":
                view = frame.copy()
                if self.calib_corners is not None:
                    cv2.drawChessboardCorners(view, calib.INNER,
                                              self.calib_corners, True)
            elif task in ("board", "mission"):
                view = self.est.draw(frame, pose, corners, ids)
            else:
                view = frame.copy()
                if corners is not None and ids is not None:
                    cv2.aruco.drawDetectedMarkers(view, corners, ids)
            if self.show_hud:
                hud = f"{mode}  {task}  rc {rc}  {fps:4.1f} fps"
                if self.downvision:
                    hud += "  [DOWN CAM]"
                cv2.putText(view, hud, (12, view.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(view, hud, (12, view.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            self._full_view = view              # full size, for the recorder
            view = cv2.resize(view, (VIDEO_W, VIDEO_H))

        yaw = pose.yaw_deg if pose is not None else 0.0
        extra = (f"{mode}   rms "
                 + ("--" if pose is None
                    else f"{pose.reproj_rms_px:.2f} px   n {pose.n_markers}")
                 + "   click map: set point")
        sp = self.setpoint
        mp = self.lmap.render(MAP_W, MAP_H, p, yaw, trail,
                              seen=pose is not None,
                              hud=self.lmap.hud_lines(p, yaw, extra, setpoint=sp),
                              setpoint=sp, limits=self.sp_limits)
        return view, mp

    # -- 3D ------------------------------------------------------------
    def render3d(self, room_map):
        with self.state_lock:
            rp = self.room_pose
        trail = list(self.room_trail)
        if room_map is None:
            img = np.full((VIEW3D_H, VIEW3D_W, 3), 26, np.uint8)
            cv2.putText(img, "no map yet", (VIEW3D_W // 2 - 70,
                                            VIEW3D_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 160), 2)
            return img
        n = len(room_map)
        hud = [f"{n} marker(s)"
               + ("  + board" if room_map.has_board else "  no board")
               + "   drag to orbit, wheel to zoom"]
        return self.view3d.render(VIEW3D_W, VIEW3D_H, room_map,
                                  p_world_cm=None if rp is None else rp.p_world_cm,
                                  R_world_cam=None if rp is None else rp.R_world_cam,
                                  trail=trail, seen=rp is not None, hud=hud)

    # -- tab text ------------------------------------------------------
    def roommap_text(self):
        m = self.mapper
        if not m.seen_count:
            return (f"frames {m.frames_used}   no room markers yet (ids 4+)\n"
                    f"marker size {self.args.marker_size * 100:.0f} cm")
        links = "  ".join(
            f"{'bd' if i < 0 else i}:{m.seen_count[i]}/{len(m.edges[i])}"
            for i in sorted(m.seen_count))
        lines = [f"frames {m.frames_used}    seen/links   {links}"]
        if m.include_board and -1 not in m.seen_count:
            lines.append("board NOT seen - no coarse approach until it "
                         "shares a frame")
        elif self.draft_map is None:
            lines.append("nothing connected yet - keep a seen marker in "
                         "frame as a new one appears")
        return "\n".join(lines)

    def roomloc_text(self):
        if self.localizer is None:
            return f"no map loaded ({self.args.map})"
        with self.state_lock:
            rp = self.room_pose
        if self.task != "roomloc":
            return str(self.room_map)
        if rp is None:
            return f"{self.room_map}\nnot localized"
        x, y, z = rp.p_world_cm
        return (f"x {x:+7.1f}  y {y:+7.1f}  z {z:+7.1f} cm\n"
                f"yaw {rp.yaw_deg:+6.1f}  pitch {rp.pitch_deg:+6.1f}  "
                f"roll {rp.roll_deg:+6.1f}\n"
                f"rms {rp.reproj_rms_px:.2f} px   n {rp.n_markers}   "
                f"ids {rp.ids}")

    def calib_text(self):
        seen = self.calib_corners is not None
        lines = [f"shots: {self.calib_shots}   (want 20-30)",
                 f"coverage: {len(self.calib_cells)}/9 regions",
                 f"board in view: {'YES' if seen else 'no'}", ""]
        r = self.calib_result
        if r:
            fx, fy = r["fov"]
            sd = r.get("std", [0, 0, 0, 0])
            lines += [f"RMS  {r['rms']:.3f} px   (want < 0.5)",
                      f"used {r['n']}/{r['total']} frames",
                      f"worst frame {r['worst']:.3f} px",
                      f"fx {r['K'][0,0]:.1f} +-{sd[0]:.1f}  fy {r['K'][1,1]:.1f}",
                      f"cx {r['K'][0,2]:.1f}  cy {r['K'][1,2]:.1f}",
                      f"FOV {fx:.1f} x {fy:.1f} deg",
                      "     (Tello stream is roughly 55 x 43)", ""]
            probs = r.get("problems") or []
            lines += (["NOT GOOD ENOUGH TO FLY ON:"]
                      + [f" - {p[:44]}" for p in probs] if probs
                      else ["passes the checks - now verify", "with a tape measure"])
        else:
            lines += ["tilt 20-45 deg in varied directions,",
                      "fill 1/3-2/3 of the frame, and cover",
                      "the corners - not just dead centre."]
        return "\n".join(lines)

    # -- Tk -----------------------------------------------------------
    def build(self):
        self.root = tk.Tk()
        self.root.title("Tello precision landing"
                        + ("" if self.live else "  [no drone]"))
        self.root.configure(bg="#1b1b1f")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", self.on_key)

        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.mode_var = tk.StringVar(value="IDLE")

        ttk.Button(bar, text="Takeoff", command=self.cmd_takeoff
                   ).pack(side="left", padx=3)
        ttk.Button(bar, text="Land", command=self.cmd_land
                   ).pack(side="left", padx=3)
        ttk.Button(bar, text="Hover", command=self.cmd_hover
                   ).pack(side="left", padx=3)
        tk.Button(bar, text="EMERGENCY", bg="#a11", fg="white",
                  activebackground="#d22", command=self.cmd_emergency
                  ).pack(side="left", padx=10)
        # Records what the window shows (camera + side panel, or camera
        # only - right-click the video) to recordings/, with a CSV log.
        self.rec_btn = tk.Button(bar, text="● Rec", width=11, bg="#2a2a30",
                                 fg="#ff5555", activebackground="#444",
                                 command=self.cmd_record)
        self.rec_btn.pack(side="left", padx=(0, 6))

        ttk.Label(bar, text="mode").pack(side="left", padx=(14, 3))
        for m in ("IDLE", "MANUAL", "HOLD"):
            ttk.Radiobutton(bar, text=m, value=m, variable=self.mode_var,
                            command=lambda m=m: self.set_mode(m)
                            ).pack(side="left")

        # AUTO HOLD setpoint, cm in the board's level frame. Also set by
        # clicking the Board tab's map; Enter (or the arrows) applies.
        ttk.Label(bar, text="setpoint cm").pack(side="left", padx=(14, 3))
        lim = self.sp_limits
        ranges = ((-150, 150), (-60, lim.y_max), (lim.z_min, lim.z_max))
        self.sp_vars = []
        for name, (lo, hi), val in zip("xyz", ranges, self.setpoint):
            ttk.Label(bar, text=name).pack(side="left", padx=(4, 1))
            var = tk.StringVar(value=f"{val:.0f}")
            sb = ttk.Spinbox(bar, from_=lo, to=hi, increment=5, width=5,
                             textvariable=var, command=self._on_sp_entry)
            sb.bind("<Return>", self._on_sp_entry)
            sb.bind("<FocusOut>", self._on_sp_entry)
            sb.pack(side="left")
            self.sp_vars.append(var)
        ttk.Button(bar, text="Reset", width=6, command=self.cmd_sp_reset
                   ).pack(side="left", padx=(4, 0))

        ttk.Label(bar, text="speed").pack(side="left", padx=(14, 3))
        self.speed_var = tk.IntVar(value=self.speed)
        ttk.Scale(bar, from_=10, to=100, variable=self.speed_var, length=90,
                  command=self._on_speed).pack(side="left")
        self.speed_lbl = ttk.Label(bar, text=f"{self.speed}")
        self.speed_lbl.pack(side="left", padx=3)

        self.video_lbl = tk.Label(self.root, bg="#000")
        self.video_lbl.grid(row=1, column=0, padx=6, pady=4)

        self.ctx = tk.Menu(self.root, tearoff=0)
        self.overlay_var = tk.StringVar(value=self.overlay)
        for label, val in (("Full overlay (pose + markers)", "full"),
                           ("Markers only", "markers"),
                           ("Normal camera (no overlay)", "clean")):
            self.ctx.add_radiobutton(label=label, value=val,
                                     variable=self.overlay_var,
                                     command=self._set_overlay)
        self.ctx.add_separator()
        self.hud_var = tk.BooleanVar(value=self.show_hud)
        self.ctx.add_checkbutton(label="Show status line",
                                 variable=self.hud_var, command=self._set_hud)
        self.down_var = tk.BooleanVar(value=False)
        self.ctx.add_checkbutton(label="Downward camera (kills pose)",
                                 variable=self.down_var,
                                 command=self.cmd_toggle_downvision)
        self.ctx.add_separator()
        self.ctx.add_command(label="Save snapshot", command=self.cmd_snapshot)
        self.ctx.add_command(label="Start / stop recording", command=self.cmd_record)
        self.rec_panel_var = tk.BooleanVar(value=self.rec_with_panel)
        self.ctx.add_checkbutton(
            label="Recording includes the side panel (off: camera only,"
                  " replays in tello_map.py)",
            variable=self.rec_panel_var, command=self._set_rec_panel)
        self.video_lbl.bind("<Button-3>", self._show_ctx)
        self.video_lbl.bind("<Button-2>", self._show_ctx)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=1, column=1, padx=6, pady=4, sticky="n")
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab)

        # Board: the localization map. Click or drag on it to place the
        # AUTO HOLD setpoint: top-down panel -> x, z; side panel -> z, y.
        board_tab = ttk.Frame(self.tabs)
        self.map_lbl = tk.Label(board_tab, bg="#000", cursor="crosshair")
        self.map_lbl.pack()
        self.map_lbl.bind("<Button-1>", self._on_map_click)
        self.map_lbl.bind("<B1-Motion>", self._on_map_click)
        self.map_lbl.bind("<ButtonRelease-1>",
                          lambda ev: self._on_map_click(ev, release=True))
        self.map_lbl.bind("<MouseWheel>", self._on_map_wheel)
        bv = ttk.Frame(board_tab)
        bv.pack(anchor="w", pady=(4, 0))
        self.board_view_var = tk.StringVar(value=self.board_view)
        ttk.Radiobutton(bv, text="2D map (click: setpoint)", value="2d",
                        variable=self.board_view_var,
                        command=self._set_board_view).pack(side="left", padx=4)
        ttk.Radiobutton(bv, text="3D view (drag: orbit)", value="3d",
                        variable=self.board_view_var,
                        command=self._set_board_view).pack(side="left", padx=4)
        ttk.Button(bv, text="Re-frame 3D", command=self.frame_board3d
                   ).pack(side="left", padx=4)
        self.tabs.add(board_tab, text="Board")

        # Mission: waypoint markers, then the board and the landing
        ms = ttk.Frame(self.tabs, padding=4)
        self.mission_lbl = tk.Label(ms, bg="#000")
        self.mission_lbl.pack()
        mrow = ttk.Frame(ms)
        mrow.pack(anchor="w", pady=(6, 0))
        ttk.Label(mrow, text="waypoint markers, in order:").pack(side="left")
        self.wp_var = tk.StringVar(value=self.args.waypoints)
        ttk.Entry(mrow, textvariable=self.wp_var, width=10).pack(side="left", padx=4)
        self.mission_hold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ms, text="hold 70 cm from the board (no hop, no landing)",
                        variable=self.mission_hold_var).pack(anchor="w")
        self.mission_rec_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ms, text="record video + log to recordings/",
                        variable=self.mission_rec_var).pack(anchor="w")
        mb = ttk.Frame(ms)
        mb.pack(anchor="w", pady=(4, 0))
        tk.Button(mb, text="Start mission...", bg="#264", fg="white",
                  activebackground="#386", command=self.cmd_mission_start
                  ).pack(side="left", padx=(0, 6))
        tk.Button(mb, text="Stop mission (land)", bg="#a11", fg="white",
                  activebackground="#d22", command=self.cmd_mission_stop
                  ).pack(side="left")
        self.mission_var = tk.StringVar(value="")
        ttk.Label(ms, textvariable=self.mission_var, font=("Consolas", 9)
                  ).pack(anchor="w", pady=(4, 0))
        self.tabs.add(ms, text="Mission")

        # Room map: 3D view of the map as it is built
        rm = ttk.Frame(self.tabs, padding=4)
        self.roommap_lbl = tk.Label(rm, bg="#000")
        self.roommap_lbl.pack()
        self._bind_orbit(self.roommap_lbl)
        self.roommap_var = tk.StringVar()
        ttk.Label(rm, textvariable=self.roommap_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))
        rmb = ttk.Frame(rm)
        rmb.pack(anchor="w", pady=6)
        ttk.Button(rmb, text="Save Map", command=self.cmd_save_map
                   ).pack(side="left", padx=2)
        ttk.Button(rmb, text="Reset", command=self.cmd_reset_map
                   ).pack(side="left", padx=2)
        ttk.Button(rmb, text="Re-frame", command=self.cmd_reframe
                   ).pack(side="left", padx=2)
        self.tabs.add(rm, text="Room map")

        # Room localization: the same 3D view, against the saved map
        rl = ttk.Frame(self.tabs, padding=4)
        self.roomloc_lbl = tk.Label(rl, bg="#000")
        self.roomloc_lbl.pack()
        self._bind_orbit(self.roomloc_lbl)
        self.roomloc_var = tk.StringVar()
        ttk.Label(rl, textvariable=self.roomloc_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))
        rlb = ttk.Frame(rl)
        rlb.pack(anchor="w", pady=6)
        ttk.Button(rlb, text="Reload Map", command=self._load_room_map
                   ).pack(side="left", padx=2)
        ttk.Button(rlb, text="Open 3D View", command=self.cmd_open_3d
                   ).pack(side="left", padx=2)

        self.tabs.add(rl, text="Room loc")

        # Calibration
        cb = ttk.Frame(self.tabs, padding=8)
        self.calib_var = tk.StringVar()
        ttk.Label(cb, textvariable=self.calib_var, justify="left",
                  font=("Consolas", 10)).pack(anchor="w")
        cbb = ttk.Frame(cb)
        cbb.pack(anchor="w", pady=8)
        ttk.Button(cbb, text="Snap", command=self.cmd_calib_snap
                   ).pack(side="left", padx=2)
        ttk.Button(cbb, text="Solve", command=self.cmd_calib_solve
                   ).pack(side="left", padx=2)
        ttk.Button(cbb, text="Reset count", command=self.cmd_calib_clear
                   ).pack(side="left", padx=2)
        self.tabs.add(cb, text="Calibrate")

        self.telem_var = tk.StringVar(value="-")
        ttk.Label(self.root, textvariable=self.telem_var,
                  font=("Consolas", 10)).grid(row=2, column=0, columnspan=2,
                                              sticky="w", padx=10)

        self.text = tk.Text(self.root, height=6, bg="#141418", fg="#ccc",
                            font=("Consolas", 9), relief="flat")
        self.text.grid(row=3, column=0, columnspan=2, sticky="ew",
                       padx=6, pady=(4, 6))
        self.root.columnconfigure(0, weight=1)

        keys = ("keys: w/s fwd-back  a/d left-right  r/f up-down  q/e yaw  "
                "SPACE hover  t takeoff  l land  x EMERGENCY  ESC land")
        ttk.Label(self.root, text=keys, foreground="#888").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        if not self.real_calib:
            self.log("nominal intrinsics - pose scale approximate, "
                     "AUTO HOLD disabled")
        if not self.live:
            self.log("no-drone mode: flight controls inactive")
        return self.root

    # -- video panel context menu ---------------------------------------
    def _show_ctx(self, ev):
        try:
            self.ctx.tk_popup(ev.x_root, ev.y_root)
        finally:
            self.ctx.grab_release()

    def _set_overlay(self):
        self.overlay = self.overlay_var.get()
        self.log(f"overlay -> {self.overlay}")

    def _set_hud(self):
        self.show_hud = bool(self.hud_var.get())

    # -- recording ------------------------------------------------------
    def cmd_record(self):
        """Rec button: start, or stop and say where it went."""
        if self.rec is None:
            self.rec_start()
        else:
            self.rec_stop()

    def rec_start(self, folder="recordings"):
        try:
            self.rec = Recorder(folder, fps=UI_HZ, with_panel=self.rec_with_panel)
        except Exception as e:
            self.log(f"cannot record: {e}")
            self.rec = None
            return
        what = ("camera + side panel" if self.rec_with_panel else
                "camera only (replays in tello_map.py)")
        self.log(f"recording {what} -> {self.rec.video_path}")
        self._rec_button()

    def rec_stop(self):
        if self.rec is None:
            return
        rec, self.rec = self.rec, None
        self.log(f"recording saved: {rec.close()}")
        self._rec_button()

    def _rec_row(self):
        """One CSV row of what the window knows right now (after 't')."""
        nan = float("nan")
        with self.state_lock:
            pose, p, mode, task = self.pose, self.p_filt, self.mode, self.task
            rp, rc = self.room_pose, self.rc
        p = (nan,) * 3 if p is None else p
        r = (nan,) * 3 if rp is None else rp.p_world_cm
        t = self.telem or {}
        return [mode, task, *(f"{v:.1f}" for v in p),
                "" if pose is None else f"{pose.yaw_deg:.1f}",
                "" if pose is None else f"{pose.reproj_rms_px:.2f}",
                0 if pose is None else pose.n_markers,
                "" if pose is None else pose.source,
                *rc, *(f"{v:.1f}" for v in self.setpoint),
                t.get("bat", ""), t.get("h", ""), t.get("tof", ""), t.get("yaw", ""),
                *(f"{v:.1f}" for v in r)]

    def _rec_step(self):
        try:
            self.rec.write(self._full_view, self._right, self._rec_row())
        except Exception as e:
            self.log(f"recording stopped: {e}")
            self.rec_stop()
            return
        self._rec_button()

    def _rec_button(self):
        btn = getattr(self, "rec_btn", None)
        if btn is None:
            return
        if self.rec is None:
            btn.config(text="● Rec", bg="#2a2a30", fg="#ff5555")
        else:
            s = int(self.rec.seconds)
            btn.config(text=f"■ Stop {s // 60}:{s % 60:02d}", bg="#a11", fg="white")

    def _set_rec_panel(self):
        if self.rec is not None:        # the frame size is fixed per recording
            self.rec_panel_var.set(self.rec_with_panel)
            self.log("stop the recording to change what it includes")
            return
        self.rec_with_panel = bool(self.rec_panel_var.get())

    def cmd_snapshot(self):
        with self.state_lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return
        pathlib.Path("recordings").mkdir(exist_ok=True)
        fn = str(pathlib.Path("recordings") /
                 f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_snap.png")
        cv2.imwrite(fn, frame)
        self.log(f"saved {fn}")

    def cmd_toggle_downvision(self):
        """Swap the forward camera for the downward 320x240 greyscale one.

        Undocumented SDK command, and the Tello streams one camera at a
        time - so this kills board and room pose while it is on. Refused
        in AUTO HOLD, which would be servoing on a pose that just vanished.
        """
        want = bool(self.down_var.get())
        if not self.live:
            self.down_var.set(False)
            self.log("downward camera needs --live")
            return
        if self.mode == "HOLD":
            self.down_var.set(self.downvision)
            self.log("leave AUTO HOLD before swapping cameras")
            return
        try:
            self._send(f"downvision {1 if want else 0}")
        except Exception as e:
            self.down_var.set(self.downvision)
            self.log(f"downvision failed: {e}")
            return
        self.downvision = want
        self.pf.reset()
        self.log("downward camera ON - no pose while it is"
                 if want else "forward camera restored")

    def _bind_orbit(self, widget):
        """Drag to orbit, wheel to zoom - the only way to tell a map that
        is genuinely L-shaped from one that just looks that way head-on."""
        widget.bind("<ButtonPress-1>", self._orbit_press)
        widget.bind("<B1-Motion>", self._orbit_drag)
        widget.bind("<MouseWheel>", self._orbit_wheel)          # Windows/macOS
        widget.bind("<Button-4>", lambda e: self.view3d.zoom(0.9))   # X11
        widget.bind("<Button-5>", lambda e: self.view3d.zoom(1.1))

    def _orbit_press(self, ev):
        self._drag = (ev.x, ev.y)

    def _orbit_drag(self, ev):
        x0, y0 = getattr(self, "_drag", (ev.x, ev.y))
        self.view3d.orbit(-(ev.x - x0) * 0.6, (ev.y - y0) * 0.6)
        self._drag = (ev.x, ev.y)

    def _orbit_wheel(self, ev):
        self.view3d.zoom(0.9 if ev.delta > 0 else 1.1)

    def cmd_reframe(self):
        self.view3d._user_moved = False
        self.view3d.frame_map(self.draft_map or self.room_map, force=True)

    def _on_tab(self, _=None):
        names = ["board", "mission", "roommap", "roomloc", "calib"]
        try:
            task = names[self.tabs.index(self.tabs.select())]
        except Exception:
            return
        with self.state_lock:
            self.task = task
        if task == "calib":
            self.calib_corners = None
        self.log(f"tab -> {task}")

    def _on_speed(self, _=None):
        self.speed = int(self.speed_var.get())
        self.speed_lbl.config(text=f"{self.speed}")

    def tick(self):
        if self.stop.is_set():
            return
        view, mp = self.compose()
        mission_map = None
        if self.mission is not None and self.mission[0].latest is not None:
            # the mission's camera view and its map (what it records, too)
            img = self.mission[0].latest
            self._full_view = img[:, :960]
            view = cv2.resize(self._full_view, (VIDEO_W, VIDEO_H))
            mission_map = cv2.resize(img[:, 960:], (MAP_W, MAP_H))
        self._photo[0] = to_photo(view)
        self.video_lbl.config(image=self._photo[0])
        right = None
        if self.task == "mission":
            right = mission_map if mission_map is not None else mp
            self._photo[1] = to_photo(right)
            self.mission_lbl.config(image=self._photo[1])
            lander = self.mission[0] if self.mission else None
            self.mission_var.set(
                "" if lander is None else
                f"{getattr(lander, 'state', '')}  target "
                f"{'board' if lander.target == 'board' else 'marker ' + str(lander.target)}")
        elif self.task == "board":
            right = mp if self.board_view == "2d" else self.render_board3d()
            self._photo[1] = to_photo(right)
            self.map_lbl.config(image=self._photo[1])
        elif self.task == "roommap":
            right = self.render3d(self.draft_map)
            self._photo[1] = to_photo(right)
            self.roommap_lbl.config(image=self._photo[1])
            self.roommap_var.set(self.roommap_text())
        elif self.task == "roomloc":
            right = self.render3d(self.room_map)
            self._photo[1] = to_photo(right)
            self.roomloc_lbl.config(image=self._photo[1])
            self.roomloc_var.set(self.roomloc_text())
        else:
            self.calib_var.set(self.calib_text())
        self._right = right
        if self.rec is not None:
            self._rec_step()

        t = self.telem
        self.telem_var.set(
            f"bat {t.get('bat','--')}%   h {t.get('h','--')}cm   "
            f"tof {t.get('tof','--')}cm   att {t.get('pitch','--')}/"
            f"{t.get('roll','--')}/{t.get('yaw','--')}   "
            f"v {t.get('vgx','--')},{t.get('vgy','--')},{t.get('vgz','--')}   "
            f"rc {self.rc}   {'AIRBORNE' if self.airborne else 'grounded'}")

        drained = False
        while True:
            try:
                self.text.insert("end", self.msgs.get_nowait() + "\n")
                drained = True
            except queue.Empty:
                break
        if drained:
            self.text.see("end")

        self.root.after(int(1000 / UI_HZ), self.tick)

    def close(self):
        if self.stop.is_set():
            return
        if self.rec is not None:        # the window is going: say where it went
            rec, self.rec = self.rec, None
            print(f"recording saved: {rec.close()}")
        if self.mission is not None:
            self.mission[1].set()           # it lands, records 5 s, then returns
            t_end = time.time() + 30
            while self.mission is not None and time.time() < t_end:
                time.sleep(0.1)
        self.stop.set()
        try:
            if self.live and self.tello is not None:
                self._rc(0, 0, 0, 0)
                if self.airborne:
                    self._send("land", timeout=20)
                self.tello.streamoff()
                self.tello.end()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        if self.live:
            self.connect()
        self.build()
        threading.Thread(target=self._worker, daemon=True).start()
        if self.args.autoquit:
            self.root.after(int(self.args.autoquit * 1000), self.close)
        self.tick()
        self.root.mainloop()

    # -- headless smoke test -------------------------------------------
    def selftest(self, n, save=None):
        """Run the worker pipeline for n frames with no Tk, and optionally
        save the composited panels. Cheap way to check the whole chain."""
        if self.live and self.tello is None:
            self.connect()                  # --live: still no takeoff
        threading.Thread(target=self._worker, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 60:
            with self.state_lock:
                got = self.n_frames
            if got >= n:
                break
            time.sleep(0.02)
        view, mp = self.compose()
        with self.state_lock:
            pose, fps = self.pose, self.fps
        self.stop.set()
        time.sleep(0.2)
        print(f"frames {got}  pipeline {fps:.1f} fps  "
              f"pose {'none' if pose is None else pose}")
        if save:
            h = max(view.shape[0], mp.shape[0])
            canvas = np.full((h, view.shape[1] + mp.shape[1], 3), 26, np.uint8)
            canvas[:view.shape[0], :view.shape[1]] = view
            canvas[:mp.shape[0], view.shape[1]:] = mp
            cv2.imwrite(save, canvas)
            print(f"wrote {save}")
        return pose is not None


def main():
    ap = argparse.ArgumentParser(description="GUI for the landing rig")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--live", action="store_true",
                     help="connect to the Tello (the default)")
    src.add_argument("--video", help="replay a recorded file instead")
    ap.add_argument("--calib", default="tello_calib.npz")
    ap.add_argument("--calib-dir", default="calib",
                    help="where the Calibrate tab keeps its frames")
    ap.add_argument("--map", default="room_map.npz",
                    help="room map: loaded by the Room loc tab, written by "
                         "the Room map tab's Save")
    ap.add_argument("--marker-size", type=float, default=0.15,
                    help="room / waypoint marker edge length, metres")
    ap.add_argument("--waypoints", default="4 5",
                    help="the Mission tab's waypoint markers to start with")
    ap.add_argument("--standoff-far", type=float, default=None,
                    help="default: scaled to the target (board_config.py)")
    ap.add_argument("--standoff-near", type=float, default=None)
    ap.add_argument("--autoquit", type=float,
                    help="close after N seconds (smoke test)")
    ap.add_argument("--selftest", type=int, metavar="N",
                    help="run N frames headless, no window")
    ap.add_argument("--save", metavar="PNG", help="with --selftest, save panels")
    args = ap.parse_args()

    if not (args.live or args.video):
        args.live = True

    app = App(args)
    if args.selftest:
        ok = app.selftest(args.selftest, args.save)
        raise SystemExit(0 if ok else 1)
    app.run()


if __name__ == "__main__":
    main()
