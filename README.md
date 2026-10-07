# Tello ArUco precision landing

A DJI Tello visits waypoint markers, finds a 4-marker ArUco board, works out
where it is relative to the board, and lands on the pad in front of it.

The Tello's camera looks forward and can't see the pad, so the board stands
at the far edge of the pad. The drone settles in front of the board, then
makes one blind hop forward over the pad and lets the firmware's `land` do
the descent.

## Files

| File | What it does |
|---|---|
| `tello_gui.py` | **The GUI**: camera, Board map, **Mission** tab, room map/localization, calibration, recording. |
| `tello_aruco_landing.py` | **The mission**: waypoints, search, approach, settle, hop, land. Run from the GUI's Mission tab or the command line. |
| `tello_pose.py` | **Localization**: board (or marker) pose, mirror-pose tracker (image, gravity, IMU heading), Kalman filter. |
| `tello_map.py` | The localization map (top-down and side view). |
| `board_config.py` | Which board is used (default: the printed 2x2 board, ids 0-3, 70 mm). |
| `calibrate_camera.py` | Camera calibration, writes `tello_calib.npz`. |
| `tello_io.py` | Tello connection and video (RGB to BGR, startup placeholder). |
| `map_room.py`, `room_map.py`, `room_pose.py`, `room_view.py`, `view_room_3d.py` | Room map and room localization (GUI tabs). |
| `tello_cli.py` | Send Tello commands by hand. |
| `print/` | The board, the waypoint markers (`room_marker_04/05_A4.pdf`, 150 mm) and the calibration chessboard. |

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
- Waypoint markers 4 and 5 (150 mm): flat, upright, TOP mark up, centre
  at about the drone's flying height (~1 m). The drone flies level with
  each marker's centre.
- The route is fixed: find marker 4 (turning right), turn **right** to find
  marker 5, turn **left** to find the board. Searching only turns on the spot,
  so from 100 cm in front of marker 4 the drone must see marker 5 by turning
  right, and from marker 5 the board by turning left. (`WAYPOINTS` and
  `SEARCH_TURN` at the top of `tello_aruco_landing.py`.)

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

**From the GUI** (`python tello_gui.py`): open the **Mission** tab, type the
waypoints (`4 5`, already filled in), press **Start mission...** and confirm.

- With "record video + log" ticked, it saves `recordings/<time>_mission.mp4`
  and `.csv`. The recording includes the hop, the landing and 5 s after it.
- **Stop mission** (or ESC / L) lands at once. x cuts the motors (the drone drops).
- The drone stays connected afterwards. If you took off by hand first, the
  mission starts from the air.

**From the command line:**

```bash
python tello_aruco_landing.py --view --record flight.mp4 --log flight.csv
python tello_aruco_landing.py --view --hold      # stop 70 cm from the board, don't land
```

```
takeoff
  for each waypoint (in order):
    -> SEARCH    turn on the spot until that marker is seen
    -> GOTO      fly to 100 cm in front of it, settle
    -> HOVER     hold there 2 s
  -> SEARCH    turn on the spot until the board is seen
  -> APPROACH  fly to 150 cm in front of the board, settle
  -> CLOSE     fly to 70 cm in front of it, settle
  -> HOP       one 'go' forward over the pad
  -> LAND
```

It lands by itself if the battery drops below 15%, after 2 minutes (plus
45 s per waypoint), if a marker or the board isn't found within one full
turn, if it can't settle, or if the hop it computes looks wrong. There is
no obstacle sensing: keep the paths clear. Settings (distances, hover time,
tolerances, gains) are at the top of `tello_aruco_landing.py`.

The previous version of every file is in `backup_full_version_2026-10-02.zip`.
