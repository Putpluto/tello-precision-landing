# Tello ArUco precision landing

A DJI Tello visits waypoint markers 4, 5, 6 and 7, finds a 4-marker ArUco
board, works out where it is relative to the board, and lands on the pad in
front of it.

The Tello's camera looks forward and can't see the pad, so the board stands
at the far edge of the pad. The drone settles in front of the board, then
makes one blind hop forward over the pad and lets the firmware's `land` do
the descent.

## Files

| File | What it does |
|---|---|
| `tello_gui.py` | **The GUI**: camera, Board map, **Mission** tab, room map/localization, calibration, recording. |
| `tello_aruco_landing.py` | **Runs the mission** on the drone: video, localization of the current target, the log. Run from the GUI's Mission tab or the command line. The route is at the top. |
| `landing_control.py` | **The mission and the control law**, no I/O: states, standoffs, tolerances, timeouts, gains (`MissionConfig`, `default_gains`). |
| `tello_pose.py` | **Localization**: board (or marker) pose, mirror-pose tracker (image, gravity, IMU heading), Kalman filter. |
| `tello_map.py` | The localization map (top-down and side view); replays a mission's `--log`. |
| `board_config.py` | Which board is used (default: the printed 2x2 board, ids 0-3, 70 mm). |
| `calibrate_camera.py` | Camera calibration, writes `tello_calib.npz`. |
| `tello_io.py` | Tello connection and video (RGB to BGR, startup placeholder). |
| `map_room.py`, `room_map.py`, `room_pose.py`, `room_view.py`, `view_room_3d.py` | Room map and room localization (GUI tabs). |
| `tello_cli.py` | Send Tello commands by hand. |
| `tests/` | Tests, no drone needed - including the whole mission flown on a synthetic Tello (`tests/sim.py`). |
| `print/` | The board, the waypoint markers (`room_marker_04..07_A4.pdf`, 150 mm) and the calibration chessboard. |

```bash
pip install -r requirements.txt
```

## 1. Set up the board and the markers

- Print `print/aruco_landing_board_A4.pdf` at 100% and **measure one black
  square**: it must be 70 mm (otherwise `python board_config.py board --marker-mm <size>`).
- Mount it flat and upright with **id 0 at the top left**. Upside down,
  the pose comes out mirrored; the code refuses an upside-down board and
  says so.
- Board centre 40 cm above the pad, pad centre 25 cm out in front.
- Waypoint markers 4, 5, 6 and 7 (150 mm): flat, upright, TOP mark up,
  centre at about the drone's flying height (~1 m). The drone flies level
  with each marker's centre and stops 100 cm in front of it.
- **Searching only turns on the spot.** From 100 cm in front of each
  marker, the next one must be visible by turning - and from marker 7, the
  board. The drone stays at marker 7's height while it looks for the
  board, and the camera sees only about 21 degrees up and down: with the
  board 60 cm lower than marker 7, keep it **at least ~1.8 m away** from
  marker 7's stop, or mount marker 7 lower.
- Set which way to turn for each search (a full turn finds it either way;
  the right direction is just quicker): `SEARCH_TURN` at the top of
  `tello_aruco_landing.py`, `--turns` on the command line, or the GUI's
  "search turns" box. Default: right for each marker, left for the board.

## 2. Calibrate the camera

GUI **Calibrate** tab, or:

```bash
python tello_pose.py --live --snap     # press 's' about 25 times
python calibrate_camera.py             # writes tello_calib.npz and checks it
```

Vary distance (0.5-2 m), tilt and position in the picture. The mission
refuses to fly on a calibration that fails the checks.

## 3. Check the localization (no takeoff)

```bash
python tello_gui.py                       # Board tab: map of where the drone is
python tello_pose.py --live --marker 4    # a waypoint marker, from the command line
```

## 4. Fly the mission

**From the GUI** (`python tello_gui.py`): open the **Mission** tab, check
the waypoints (`4 5 6 7`, already filled in) and the search turns (e.g.
`R R L R L` - one per marker, then the board; empty for the defaults),
press **Start mission...** and confirm.

- With "record video + log" ticked, it saves `recordings/<time>_mission.mp4`
  and `.csv`. The recording includes the hop, the landing and 5 s after it.
- **Stop mission** (or ESC / L) lands at once. x cuts the motors (the drone drops).
- **Nudge the yaw by hand:** Q / E, the arrow keys, or the ◀ yaw / yaw ▶
  buttons (hold to keep turning). Each press turns at rc 25 for 0.4 s,
  overriding the mission's yaw; let go and the mission takes over again.
  On the command line (`--view` window) it is A / D or the arrow keys.
  (`NUDGE_RC`, `NUDGE_S` in `tello_aruco_landing.py`.)
- **Skip marker (N):** gives up on the current marker (searching for it,
  flying to it or hovering there) and searches for the next one, turning
  the way set for it. The board can't be skipped. On the command line: N
  in the `--view` window.
- The drone stays connected afterwards. If you took off by hand first, the
  mission starts from the air.

**From the command line:**

```bash
python tello_aruco_landing.py --view --record flight.mp4 --log flight.csv
python tello_aruco_landing.py --view --turns R R L R L   # search directions for your room
python tello_aruco_landing.py --view --hold              # stop 70 cm from the board, don't land
python tello_map.py --log flight.csv --video flight.mp4  # replay it afterwards
```

```
takeoff
  for each waypoint (4, 5, 6, 7 in order):
    -> SEARCH    turn on the spot until that marker is seen
    -> GOTO      fly to 100 cm in front of it - within 30 cm and 15 deg is enough
    -> HOVER     hold there 2 s
  -> SEARCH    turn on the spot until the board is seen
  -> APPROACH  fly to 150 cm in front of the board, settle
  -> CLOSE     fly to 70 cm in front of it, settle
  -> HOP       one 'go' forward over the pad
  -> LAND
```

It lands by itself if the battery drops below 15%, after 2 minutes (plus
45 s per waypoint - 5 minutes for 4 5 6 7), if a marker or the board isn't
found within one full turn, if it can't settle, or if the hop it computes
looks wrong. There is no obstacle sensing: keep the paths clear.

If it loses the board after finding it (APPROACH, CLOSE or HOLD), it does
not turn a full circle. It enters LOST and looks either side of where the
board was: left 2 s, right 4 s (back, then 2 s past), left 2 s back, and
again. When it sees the board it carries on where it was. Not back after
16 s: it lands. (`lost_sweep_s`, `lost_max_s` in `MissionConfig`.)

Settings: the route (waypoints, search turns) is at the top of
`tello_aruco_landing.py`; how it is flown (distances, hover time,
tolerances, timeouts) is `MissionConfig` and the gains are `default_gains()`,
both in `landing_control.py`. The GUI's AUTO HOLD and Mission tab use the
same ones.

The log (`--log`, or the GUI's mission recording) has one row per control
step: the filtered and raw pose, which evidence settled the mirror
ambiguity, the control errors and rc commands, the hop, and the drone's
own state packet (height, attitude, velocity, battery).

## Tests

```bash
pip install pytest
python -m pytest tests
```

No drone needed. `tests/test_lander.py` flies the whole mission - takeoff,
markers 4 to 7, the board, the hop, the landing - on a synthetic Tello
that renders the markers through the camera model, with the video 0.25 s
late; it takes about half a minute. It checks the code is wired right, not
that a real flight will be: the simulated drone is not a model of the
Tello's dynamics.

To watch that simulated flight:

```bash
python tests/fly_sim.py                    # live window, real time; q lands
python tests/fly_sim.py --record sim.mp4   # and save it
```

It shows the mission's own camera + map view (what `--view` shows on the
real drone) beside a 3D view of the simulated room and the drone's path.
