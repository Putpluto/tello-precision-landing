"""
Landing control: the mission state machine and the control law, no I/O.

tello_aruco_landing.py owns the drone, the video, the log and the window;
this module turns observations into commands. Nothing in here touches a
socket, a window or the wall clock - time arrives inside each observation -
so the whole mission can be stepped from a test with synthetic poses, and
the same code flies the real Tello, the simulator, and the GUI's AUTO HOLD.

FRAMES
------
Level frame (L), see tello_pose.py: origin at the board centre, +X right
as you face the print, +Y up, +Z out into the room, cm. The pad centre is
at (0, -pad_drop, pad_out). The coarse leg (--room-map) runs the same law
in the room map's frame, which is also +Y up.

rc channels, send_rc_control(a, b, c, d), each -100..100:
    a = right, b = forward, c = up, d = yaw clockwise

THE CONTROL LAW
---------------
One law for every phase. It never assumes the drone is square-on: the
position error is rotated into the body frame with the camera's measured
orientation, flattened onto the horizontal plane,

    forward = camera +Z with its vertical part removed
    right   = camera +X with its vertical part removed
    up      = +Y

and each axis is a PD loop - proportional on the position error, damping
on the Kalman filter's velocity along that axis. The nose is aimed at the
board the whole time (yaw error = bearing of the aim point), so arriving
at the standoff means arriving with the board centred.

    e_fwd > 0 (target ahead)     -> b > 0 (forward)
    e_lat > 0 (target right)     -> a > 0 (right)
    e_up  > 0 (target above)     -> c > 0 (climb)
    aim right of the nose        -> d > 0 (yaw clockwise)

tests/test_landing_control.py checks every one of those signs.

When the planar ambiguity is unresolved (tello_pose.BoardTracker), the
lateral error's sign is exactly what is in doubt, so the lateral axis is
held at zero; range, height and bearing are the same for both candidates
and keep working.

WHY THE OLD CONTROLLER NEVER LANDED
-----------------------------------
It advanced only after 12 consecutive frames with all four rc outputs at
exactly zero. With P-only gains of ~0.35 rc/cm, a 7 cm error produced
rc 2 - a 2 cm/s crawl that ordinary drift cancels - so it hovered just
outside the deadband until the mission timed out (all three recorded
flights, and every simulated one). Settling here is judged on the errors
and the measured speed, held for a time, not on the outputs.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


# ----------------------------------------------------------------------
# Gains
# ----------------------------------------------------------------------

@dataclass
class Gains:
    kp: float                 # rc units per cm (per degree for yaw)
    kd: float = 0.0           # rc units per cm/s along the axis (damping)
    limit: float = 30.0       # |rc| cap
    slew: float = 10.0        # max rc change per control step
    deadband: float = 0.0     # cm (deg); subtracted, so the output is continuous
    ki: float = 0.0           # rc units per cm*s: cancels a steady push (drift)
    i_zone: float = 30.0      # cm: only integrate this close to the target
    i_limit: float = 12.0     # |rc| the integral may contribute
    i_closing: float = 2.0    # cm/s: no integrating while closing faster than this


def default_gains():
    """Tuned in the simulator (rc 100 ~ 1 m/s, 0.25 s video delay), with
    margin: sim_eval.py varies the response by +-40% and the delay by
    +-0.1 s and they still land. A real pass should start here, one axis
    at a time (README, Tuning).

    The integral terms matter when holding: a PD loop on a velocity-
    commanded drone settles wherever its correction balances a steady
    drift - 5-6 cm short in the simulator at 2.5 cm/s - and the integral is
    what takes that out. Integral time kp/ki ~ 5 s, several times the
    loop's response, so it cannot fight the damping."""
    return {
        "fwd": Gains(kp=0.55, kd=0.30, limit=30.0, ki=0.10),
        "lat": Gains(kp=0.55, kd=0.30, limit=25.0, ki=0.10),
        "vert": Gains(kp=0.60, kd=0.20, limit=25.0, ki=0.10),
        "yaw": Gains(kp=0.90, kd=0.00, limit=30.0, deadband=0.5),
    }


class Axis:
    """PID on one axis with deadband, clamp and slew limit.

    The integral is for a STEADY push - drift, a mis-trimmed airframe that
    always slides one way - so it only runs when the drone is being held
    off target, not while it is getting there:
      - not while closing on the target faster than i_closing (an approach
        would otherwise charge it, and carry the drone past the target),
      - only within i_zone of the target,
      - clamped to i_limit, and frozen while the output is pinned at the
        limit in the direction it would push.
    """

    def __init__(self, g: Gains):
        self.g = g
        self.prev = 0.0
        self.integ = 0.0

    def step(self, error, rate=0.0, dt=1.0 / 15.0, integrate=True):
        """error: target minus position along the axis; rate: velocity
        along the same axis (so moving toward the target is rate with the
        sign of error)."""
        g = self.g
        e = math.copysign(max(abs(error) - g.deadband, 0.0), error)
        integ = self.integ
        closing = rate * math.copysign(1.0, error)
        if (g.ki and integrate and abs(error) < g.i_zone
                and closing < g.i_closing):
            integ = float(np.clip(integ + g.ki * e * dt, -g.i_limit, g.i_limit))
        u_raw = g.kp * e - g.kd * rate + integ
        if abs(u_raw) <= g.limit or (u_raw > 0) != (e > 0):
            self.integ = integ               # not pushing into saturation: keep it
        u = float(np.clip(g.kp * e - g.kd * rate + self.integ, -g.limit, g.limit))
        # slew limit: the Tello responds badly to step changes in rc
        u = float(np.clip(u, self.prev - g.slew, self.prev + g.slew))
        self.prev = u
        return u

    def reset(self):
        self.prev = 0.0
        self.integ = 0.0


# ----------------------------------------------------------------------
# The law
# ----------------------------------------------------------------------

def horizontal_axes(R_world_cam):
    """Body forward and right as unit vectors in the horizontal plane of a
    +Y-up frame, or (None, None) if the camera points too near straight
    up or down to define a heading. Dropping the vertical component keeps
    a pitched camera from leaking climb into the forward axis."""
    R = np.asarray(R_world_cam, float)
    fwd = R[:, 2].copy()
    right = R[:, 0].copy()
    fwd[1] = 0.0
    right[1] = 0.0
    nf, nr = np.linalg.norm(fwd), np.linalg.norm(right)
    if nf < 1e-3 or nr < 1e-3:
        return None, None
    return fwd / nf, right / nr


@dataclass
class ServoOut:
    rc: tuple
    e_fwd: float
    e_lat: float          # NaN when the pose is unresolved: its sign is unknown
    e_up: float
    e_yaw: float
    v_fwd: float
    v_lat: float
    v_up: float
    e_lat_abs: float = 0.0  # |lateral error| - known either way: the mirror
                            # candidate is the same distance off-axis

    @property
    def dist(self):
        return math.sqrt(self.e_fwd ** 2 + self.e_lat ** 2 + self.e_up ** 2)

    @property
    def speed(self):
        return math.sqrt(self.v_fwd ** 2 + self.v_lat ** 2 + self.v_up ** 2)

    def speed_known(self, resolved=True):
        """Speed from the axes whose velocity can be trusted: unresolved, the
        lateral velocity estimate may be the mirror's."""
        if resolved:
            return self.speed
        return math.sqrt(self.v_fwd ** 2 + self.v_up ** 2)


class ServoLaw:
    """Fly to `target` while pointing the nose at `aim`, both in the same
    +Y-up frame as the position p (cm) and velocity v (cm/s)."""

    def __init__(self, gains=None):
        self.gains = gains or default_gains()
        self.axes = {k: Axis(g) for k, g in self.gains.items()}

    def reset(self):
        for ax in self.axes.values():
            ax.reset()

    def step(self, p, v, R_world_cam, target, aim, lateral=True, dt=1.0 / 15.0):
        fwd_h, right_h = horizontal_axes(R_world_cam)
        if fwd_h is None:
            return None
        p = np.asarray(p, float)
        v = np.zeros(3) if v is None else np.asarray(v, float)
        e = np.asarray(target, float) - p
        e_f, e_r, e_u = float(e @ fwd_h), float(e @ right_h), float(e[1])
        v_f, v_r, v_u = float(v @ fwd_h), float(v @ right_h), float(v[1])
        b = self.axes["fwd"].step(e_f, v_f, dt)
        if lateral:
            a = self.axes["lat"].step(e_r, v_r, dt)
        else:
            # the sign of e_r (and of v_r) is what the ambiguity flips:
            # hold the axis - no integral either, it was charged on a sign
            # that is now in doubt - and let the slew limit bring it down
            self.axes["lat"].integ = 0.0
            a = self.axes["lat"].step(0.0, 0.0, dt, integrate=False)
        c = self.axes["vert"].step(e_u, v_u, dt)
        # Aim at the board, not at the target: the target's bearing goes
        # undefined exactly as you arrive, and arriving already pointed at
        # the board is what keeps it in view for the next phase.
        to_aim = np.asarray(aim, float) - p
        yaw_err = math.degrees(math.atan2(float(to_aim @ right_h),
                                          float(to_aim @ fwd_h)))
        d = self.axes["yaw"].step(yaw_err, 0.0, dt)
        rc = tuple(int(round(x)) for x in (a, b, c, d))
        return ServoOut(rc, e_f, e_r if lateral else float("nan"), e_u, yaw_err,
                        v_f, v_r, v_u, e_lat_abs=abs(e_r))


# ----------------------------------------------------------------------
# Setpoints: where the drone can hold and still see the board
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SetpointLimits:
    """The region of the level frame (cm) from which a level Tello can
    still see - and measure - the board. Outside it, holding the setpoint
    would take the board out of view and the servo loses its sensor.

    The Tello's camera cannot tilt and its vertical field of view is only
    ~43 deg, so the board has to stay within a narrow wedge above and below
    the camera's axis: max_elev_deg leaves room for the board's own size
    and the pitch the drone uses to move. Sideways, the drone yaws to face
    the board, so the limit is how obliquely the markers can be read.
    """
    z_min: float = 50.0            # closer and a wobble reaches the board
    z_max: float = 250.0           # past this the 70 mm markers are ~25 px
    max_view_deg: float = 40.0     # off the board's axis, seen from above
    max_elev_deg: float = 12.0     # above/below the camera's level axis
    floor_clear: float = 25.0      # cm above the pad surface
    y_max: float = 120.0


def clamp_setpoint(sp, limits=SetpointLimits(), pad_drop=40.0):
    """(x, y, z) cm in L -> the nearest point inside `limits`, plus a
    list of what was limited and why (empty if nothing was)."""
    x, y, z = (float(v) for v in sp)
    notes = []
    z2 = min(max(z, limits.z_min), limits.z_max)
    if abs(z2 - z) > 0.5:
        notes.append(f"range z held to {limits.z_min:.0f}-{limits.z_max:.0f} cm")
    x_max = math.tan(math.radians(limits.max_view_deg)) * z2
    x2 = min(max(x, -x_max), x_max)
    if abs(x2 - x) > 0.5:
        notes.append(f"x held to +-{x_max:.0f} cm at this range (markers stop "
                     f"reading past {limits.max_view_deg:.0f} deg off-axis)")
    y_view = math.tan(math.radians(limits.max_elev_deg)) * math.hypot(x2, z2)
    y_lo = max(-y_view, -pad_drop + limits.floor_clear)
    y_hi = min(y_view, limits.y_max)
    y2 = min(max(y, y_lo), y_hi)
    if abs(y2 - y) > 0.5:
        why = ("the board would leave the camera's view"
               if y2 == y_view or y2 == -y_view else "too close to the floor")
        notes.append(f"height y held to {y_lo:+.0f}..{y_hi:+.0f} cm here ({why}; "
                     "move further back to go higher or lower)")
    return np.array([x2, y2, z2]), notes


# ----------------------------------------------------------------------
# Mission
# ----------------------------------------------------------------------

class State(Enum):
    COARSE = "COARSE"         # room-map leg toward the standoff, board unseen
    SEARCH = "SEARCH"         # yaw sweep until the board is seen
    APPROACH = "APPROACH"     # servo to the FAR standoff
    CLOSE = "CLOSE"           # servo to the NEAR standoff
    HOLD = "HOLD"             # --hold: stay at NEAR, no hop, no landing
    HOP = "HOP"               # one blind 'go' over the pad
    LAND = "LAND"             # firmware 'land'
    DONE = "DONE"


@dataclass
class MissionConfig:
    standoff_far: float = 150.0    # cm along +Z_L, approach hold
    standoff_near: float = 70.0    # cm, last hold before the hop
    height: float = 0.0            # cm, target y in L (0 = level with board centre)
    pad_out: float = 25.0          # cm, pad centre from the board, horizontally
    cam_fwd: float = 4.0           # cm, lens ahead of the airframe centre
    # settle: every error inside tolerance, speed low, for settle_s.
    # FAR is a waypoint - lateral noise there is several cm (it grows like
    # range^4, see tello_pose.PoseFilter) - so it only has to be roughly
    # right. NEAR is where the hop is computed, and where the pose is ~20x
    # steadier.
    tol_far: tuple = (15.0, 12.0, 15.0, 8.0, 20.0)   # fwd, lat, up cm; yaw deg; speed cm/s
    tol_near: tuple = (5.0, 3.5, 6.0, 4.0, 7.0)
    settle_s: float = 0.8
    relax_after_s: float = 12.0    # CLOSE only: widen tolerances after this long...
    relax_max: float = 1.6         # ...up to this factor, so wobble cannot stall it
    close_max_s: float = 40.0      # still not settled at NEAR: land where it is
    lost_hover_s: float = 1.2      # board gone this long -> search (or coarse)
    search_rc: int = 22            # yaw sweep, rc units
    search_max_s: float = 25.0     # > one full turn at the default rate
    mission_max_s: float = 120.0
    batt_min: int = 15             # percent
    hop_speed: int = 30            # cm/s for the go command
    hop_max: float = 95.0          # refuse a hop longer than this, cm
    coarse_tol: float = 30.0       # cm: at the waypoint, board still unseen -> SEARCH
    coarse_max: float = 600.0      # cm: refuse a coarse dash longer than this
    coarse_lost_s: float = 5.0     # room fix gone this long -> land
    hold: bool = False
    # --hold with a setpoint: (x, y, z) cm in L to hold at instead of NEAR on
    # the axis. Reached from FAR directly (CLOSE exists only for the hop).
    hold_point: tuple = None


REF_SPAN_M = 0.165          # the printed 2x2 board the defaults were tuned on


def pose_range_scale(geom):
    """How far out the pose is as good as the printed board's, relative to
    it (1.0 for the board itself). Lateral noise grows ~range^4 and falls
    ~span^2 (tello_pose.PoseFilter), so equal noise sits at range ~ sqrt(span):
    a single 70 mm marker at 1 m is about the board at 1.5 m."""
    return math.sqrt(geom.span_m / REF_SPAN_M)


def for_board(geom, **cfg_kw):
    """(MissionConfig, SetpointLimits) for a landing target. A target
    smaller than the printed board brings FAR and the setpoint range in
    to where its pose is still good; a bigger one changes nothing (the
    standoffs are not what limits it then)."""
    s = min(pose_range_scale(geom), 1.0)
    cfg = MissionConfig(pad_out=geom.pad_out_m * 100.0, **cfg_kw)
    if "standoff_far" not in cfg_kw:
        cfg.standoff_far = float(round(max(cfg.standoff_near + 30.0,
                                           MissionConfig.standoff_far * s)))
    lim = SetpointLimits(z_max=float(round(max(cfg.standoff_far + 20.0,
                                               SetpointLimits.z_max * s))))
    return cfg, lim


@dataclass
class Obs:
    """Everything the mission may use, at one control step."""
    t: float                       # now, seconds (any epoch)
    p: np.ndarray = None           # drone position in L, cm, filtered, predicted to t
    v: np.ndarray = None           # velocity in L, cm/s
    pose: object = None            # latest tello_pose.BoardPose (orientation, resolved)
    room_p: np.ndarray = None      # position in the room frame, cm (coarse leg)
    room_v: np.ndarray = None
    room_R: np.ndarray = None      # camera axes in the room frame
    battery: float = None


@dataclass
class Cmd:
    rc: tuple = (0, 0, 0, 0)
    action: str = None             # None | "hop" | "land" | "done"
    hop: tuple = None              # (forward_cm, left_cm) for go_xyz_speed
    note: str = ""                 # worth printing and logging when set
    servo: ServoOut = None


class Mission:
    """The landing sequence as a state machine. step(obs) -> Cmd; the
    caller executes the command. See the module docstring for the law and
    tello_aruco_landing.py for the loop that runs this."""

    def __init__(self, cfg=None, gains=None, room_waypoint=None):
        """room_waypoint: (target_cm, aim_cm) in the room frame - the FAR
        standoff and the board centre - to open with the coarse leg."""
        self.cfg = cfg or MissionConfig()
        self.law = ServoLaw(gains)
        self.room_law = ServoLaw(gains)
        self.room_waypoint = room_waypoint
        self.state = State.COARSE if room_waypoint is not None else State.SEARCH
        self.t0 = None
        self.t_state = None
        self.t_seen = None
        self.t_room = None
        self.t_settle = None
        self.t_prev = None
        self.dt = 1.0 / 15.0              # control period, for the integrals
        self.search_dir = 1
        self.last_note = ""
        self.reason = ""

    # -- helpers ----------------------------------------------------------
    def _goto(self, state, t, note=""):
        self.state = state
        self.t_state = t
        self.t_settle = None
        self.law.reset()
        self.room_law.reset()
        return note or f"-> {state.value}"

    def _finish(self, t, why):
        """Land where it is. The land action goes out with this command,
        so the machine goes straight to DONE."""
        self.reason = why
        note = self._goto(State.DONE, t, f"LAND: {why}")
        return Cmd(action="land", note=note)

    def _settled(self, out, tol, t, resolved, scale=1.0):
        """Every error inside tolerance and slow, held for settle_s. The
        lateral check uses the distance off-axis, which the planar
        ambiguity does not hide (only which side it is on), so a drone
        close enough to the axis settles even while the side is unknown."""
        tf, tl, tu, ty, ts = (x * scale for x in tol)
        ok = (abs(out.e_fwd) < tf and abs(out.e_up) < tu and abs(out.e_yaw) < ty
              and out.speed_known(resolved) < ts and out.e_lat_abs < tl)
        if not ok:
            self.t_settle = None
            return False
        if self.t_settle is None:
            self.t_settle = t
        return t - self.t_settle >= self.cfg.settle_s

    def hop_vector(self, p, R):
        """Body-frame displacement from here to above the pad centre, cm.
        The camera is ahead of the airframe centre, and the airframe
        centre is what has to end up over the pad."""
        fwd_h, right_h = horizontal_axes(R)
        centre = np.asarray(p, float) - fwd_h * self.cfg.cam_fwd
        target = np.array([0.0, centre[1], self.cfg.pad_out])
        d = target - centre
        return float(d @ fwd_h), float(d @ right_h)

    # -- the step -----------------------------------------------------------
    def step(self, obs: Obs) -> Cmd:
        c = self.cfg
        t = obs.t
        if self.t0 is None:
            self.t0 = self.t_state = t
        if self.t_prev is not None:
            self.dt = min(max(t - self.t_prev, 0.01), 0.3)
        self.t_prev = t
        st = self.state
        if st is State.DONE:
            return Cmd(action="done")
        if st is State.LAND:                         # after the hop
            self.reason = self.reason or "landed after the hop"
            self.state = State.DONE
            return Cmd(action="land", note="land")

        if t - self.t0 > c.mission_max_s:
            return self._finish(t, f"mission timeout {c.mission_max_s:.0f} s")
        if obs.battery is not None and obs.battery < c.batt_min:
            return self._finish(t, f"battery {obs.battery}%")

        seen = obs.p is not None and obs.pose is not None
        if seen:
            self.t_seen = t
        if obs.room_p is not None:
            self.t_room = t

        if st is State.COARSE:
            return self._coarse(obs)
        if st is State.SEARCH:
            if seen:
                return Cmd(note=self._goto(State.APPROACH, t, "board acquired -> APPROACH"))
            if t - self.t_state > c.search_max_s:
                return self._finish(t, "board not found")
            return Cmd(rc=(0, 0, 0, self.search_dir * c.search_rc))
        if st in (State.APPROACH, State.CLOSE, State.HOLD):
            return self._servo(obs, seen)
        if st is State.HOP:
            if not seen:
                return self._finish(t, "board lost before the hop - landing in place")
            fwd, right = self.hop_vector(obs.p, obs.pose.R_level_cam)
            side_note = ""
            if not getattr(obs.pose, "resolved", True):
                # Which side of the axis is unknown - but how far is not,
                # and it settled inside tolerance - so straight ahead is
                # never more than that far off. Guessing the side could
                # double it.
                side_note = f" (side unknown, |x| {abs(right):.0f} cm - straight)"
                right = 0.0
            if fwd < -5.0 or fwd > c.hop_max or abs(right) > 40.0:
                return self._finish(t, f"hop ({fwd:.0f}, {right:.0f}) cm is not "
                                       "plausible - landing in place")
            self.t_state = t
            if max(abs(fwd), abs(right)) < 20.0:
                # 'go' refuses moves under 20 cm; this close, just land
                self.state = State.DONE
                self.reason = "landed (already over the pad)"
                return Cmd(action="land", note=f"over the pad already "
                                                f"({fwd:.0f}, {right:.0f}) cm - land")
            self.state = State.LAND
            return Cmd(action="hop", hop=(int(round(fwd)), int(round(-right))),
                       note=f"HOP forward {fwd:.0f} right {right:.0f} cm{side_note}")
        return Cmd()

    def _servo(self, obs, seen):
        c = self.cfg
        t = obs.t
        st = self.state
        if not seen:
            lost = t - (self.t_seen if self.t_seen is not None else self.t_state)
            if lost > c.lost_hover_s:
                if self.room_waypoint is not None:
                    return Cmd(note=self._goto(State.COARSE, t,
                                               "board lost -> COARSE (back to the waypoint)"))
                return Cmd(note=self._goto(State.SEARCH, t, "board lost -> SEARCH"))
            return Cmd(rc=(0, 0, 0, 0))              # hover through a dropout

        if st is State.HOLD and c.hold_point is not None:
            target = np.asarray(c.hold_point, float)
        else:
            standoff = c.standoff_far if st is State.APPROACH else c.standoff_near
            target = np.array([0.0, c.height, standoff])
        # nose on the board wherever the target is: that is what keeps the
        # board in view from an off-axis setpoint
        aim = np.array([0.0, c.height, 0.0])
        resolved = bool(getattr(obs.pose, "resolved", True))
        out = self.law.step(obs.p, obs.v, obs.pose.R_level_cam, target, aim,
                            lateral=resolved, dt=self.dt)
        if out is None:
            return Cmd(rc=(0, 0, 0, 0), note="no usable heading")
        # remember which way to turn if the board is lost
        if abs(out.e_yaw) > 2.0:
            self.search_dir = 1 if out.e_yaw > 0 else -1
        cmd = Cmd(rc=out.rc, servo=out)
        if st is State.APPROACH:
            if self._settled(out, c.tol_far, t, resolved):
                if c.hold and c.hold_point is not None:
                    x, y, z = c.hold_point
                    cmd.note = self._goto(State.HOLD, t, "settled at FAR -> HOLD at "
                                                         f"setpoint ({x:+.0f}, {y:+.0f}, {z:.0f}) cm")
                else:
                    cmd.note = self._goto(State.CLOSE, t, "settled at FAR -> CLOSE")
        elif st is State.CLOSE:
            in_state = t - self.t_state
            scale = 1.0 + min(max(in_state - c.relax_after_s, 0.0) / 10.0,
                              c.relax_max - 1.0)
            if self._settled(out, c.tol_near, t, resolved, scale):
                if c.hold:
                    cmd.note = self._goto(State.HOLD, t, "settled at NEAR -> HOLD (--hold)")
                else:
                    cmd.note = self._goto(State.HOP, t, "settled at NEAR -> HOP")
            elif in_state > c.close_max_s:
                return self._finish(t, "could not settle at NEAR - landing in place")
        return cmd

    def _coarse(self, obs):
        c = self.cfg
        t = obs.t
        if obs.p is not None and obs.pose is not None:
            # the precise sensor is available: hand over mid-flight
            return Cmd(note=self._goto(State.APPROACH, t, "board acquired - COARSE -> APPROACH"))
        if obs.room_p is None or obs.room_R is None:
            last = self.t_room if self.t_room is not None else self.t_state
            if t - last > c.coarse_lost_s:
                return self._finish(t, "room localization lost")
            return Cmd(rc=(0, 0, 0, 0))
        target, aim = self.room_waypoint
        out = self.room_law.step(obs.room_p, obs.room_v, obs.room_R, target, aim,
                                 dt=self.dt)
        if out is None:
            return Cmd(rc=(0, 0, 0, 0), note="no usable heading")
        if out.dist > c.coarse_max:
            return self._finish(t, f"coarse waypoint {out.dist:.0f} cm away, over "
                                   f"the {c.coarse_max:.0f} cm limit - check the room map")
        if out.dist < c.coarse_tol:
            return Cmd(note=self._goto(State.SEARCH, t, f"at the waypoint ({out.dist:.0f} cm) "
                                                        "but no board in view -> SEARCH"))
        return Cmd(rc=out.rc, servo=out)
