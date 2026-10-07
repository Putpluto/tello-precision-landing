"""
Which landing target the code uses - the printed 2x2 board (the default)
or a single marker - stored in board.json next to the code, so the landing
script, the GUI, the map and make_targets.py all agree.

    python board_config.py                              # what is configured now
    python board_config.py single --id 0 --marker-mm 70 # one marker, id 0, 70 mm
    python board_config.py board                        # back to the printed 2x2 board
    python board_config.py board --marker-mm 100 --pitch-mm 135     # an A3 print
    python board_config.py single --pad-drop-mm 380 --pad-out-mm 250 # how it is mounted
    python board_config.py identify photo.jpg           # which dictionary / id is this?
    python board_config.py identify --live              # ...seen by the Tello (no takeoff)

MEASURE the marker's black square, edge to edge - not the paper. Every
distance the drone flies is scaled by this number.

A SINGLE MARKER sits at the board origin: pad_drop is measured from its
centre down to the pad surface, pad_out from its face to the pad centre,
horizontally. It works, but one small marker gives a noisier pose than the
2x2 board (4 corners over 7 cm instead of 16 over 16.5 cm), so the approach
is scaled to it: the FAR standoff moves in to where the pose is as good as
the board's at 150 cm (landing_control.for_board). In the simulator:

    lateral error, 90th pct   0.7 m    1.0 m    1.5 m    2.0 m
    2x2 board (70 mm)         0.2 cm   0.5 cm   3 cm     6 cm
    single 70 mm marker       0.6 cm   2.8 cm   12 cm    24 cm
    single 150 mm marker      0.3 cm   0.8 cm   2 cm     8 cm

So if you can print one, a single LARGE marker (python make_targets.py
after configuring it) is as good as the board.

TELLO_BOARD in the environment overrides the file: a path to another
board.json, or "default" to ignore board.json (the tests do, so they do
not depend on how your drone is set up).
"""

import argparse
import dataclasses
import json
import os
import pathlib

import cv2

from tello_pose import BoardGeometry

PATH = pathlib.Path(__file__).resolve().parent / "board.json"
ENV = "TELLO_BOARD"

# the dictionaries worth guessing when identifying a marker of unknown origin
DICTS = {name: getattr(cv2.aruco, "DICT_" + name) for name in (
    "4X4_50", "4X4_100", "4X4_250", "4X4_1000", "5X5_50", "5X5_100", "5X5_250",
    "6X6_50", "6X6_250", "7X7_50", "ARUCO_ORIGINAL", "APRILTAG_36h11")}


def dict_name(dict_id):
    for name, v in DICTS.items():
        if v == dict_id:
            return name
    return str(dict_id)


def load_board(path=None):
    """The configured BoardGeometry: board.json if there is one, else the
    printed 2x2 board."""
    env = os.environ.get(ENV)
    if env == "default" and path is None:
        return BoardGeometry()
    p = pathlib.Path(path or env or PATH)
    if not p.exists():
        return BoardGeometry()
    d = json.loads(p.read_text())
    kw = {}
    if "ids" in d:
        kw["ids"] = tuple(int(i) for i in d["ids"])
    if "dict" in d:
        kw["dict_id"] = DICTS[d["dict"]]
    for key, field in (("marker_mm", "marker_m"), ("pitch_mm", "pitch_m"),
                       ("pad_drop_mm", "pad_drop_m"), ("pad_out_mm", "pad_out_m")):
        if key in d:
            kw[field] = float(d[key]) / 1000.0
    if "tilt_deg" in d:
        kw["tilt_deg"] = float(d["tilt_deg"])
    geom = BoardGeometry(**kw)
    geom.layout                        # raises on a malformed id list
    return geom


def save_board(geom, path=PATH):
    d = {"ids": list(geom.ids), "dict": dict_name(geom.dict_id),
         "marker_mm": round(geom.marker_m * 1000.0, 2),
         "pad_drop_mm": round(geom.pad_drop_m * 1000.0, 1),
         "pad_out_mm": round(geom.pad_out_m * 1000.0, 1),
         "tilt_deg": geom.tilt_deg}
    if not geom.single:
        d["pitch_mm"] = round(geom.pitch_m * 1000.0, 2)
    pathlib.Path(path).write_text(json.dumps(d, indent=2) + "\n")


def describe(geom):
    from landing_control import for_board
    cfg, lim = for_board(geom)
    out = [geom.check(),
           f"dictionary {dict_name(geom.dict_id)} | pad {geom.pad_drop_m*100:.0f} cm below the "
           f"target centre, {geom.pad_out_m*100:.0f} cm out | tilt {geom.tilt_deg:.0f} deg",
           f"approach: FAR {cfg.standoff_far:.0f} cm, NEAR {cfg.standoff_near:.0f} cm; "
           f"setpoints up to {lim.z_max:.0f} cm out"]
    if geom.single and geom.marker_m < 0.10:
        out.append("note: a single marker this small gives a noisy pose past ~1 m - the "
                   "approach is kept close; a bigger marker (or the 2x2 board) does better")
    if geom.single and geom.marker_m < 0.05:
        out.append("WARNING: under 5 cm the pose at the NEAR standoff is too noisy for a "
                   "precise hop - use a bigger marker")
    return "\n".join(out)


# ----------------------------------------------------------------------
# identify: what is this marker?
# ----------------------------------------------------------------------

def identify(img):
    """Every (dictionary, id, side_px) that decodes in img. A marker from
    the wrong dictionary is invisible to the detector - this is how to find
    out which one a marker from somewhere else belongs to."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = []
    for name, did in DICTS.items():
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(did),
                                      cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            side = float(cv2.arcLength(c.reshape(-1, 2).astype("float32"), True) / 4.0)
            found.append((name, int(i), side))
    return found


def _grab_frame():
    import time
    from tello_io import FrameGrabber, open_drone
    drone = open_drone()
    try:
        g = FrameGrabber(drone)
        t_end = time.time() + 8.0
        while time.time() < t_end:
            f, new = g.grab()
            if new:
                time.sleep(1.0)           # let the decoder settle, take a fresh one
                f2, _ = g.grab()
                return f2 if f2 is not None else f
            time.sleep(0.05)
        raise SystemExit("no video from the drone")
    finally:
        drone.streamoff()
        drone.end()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", nargs="?", choices=("single", "board", "identify", "reset"),
                    help="single: one marker; board: the 2x2 board; identify: which "
                         "dictionary/id is in a picture; reset: delete board.json")
    ap.add_argument("image", nargs="?", help="identify: a photo or snapshot")
    ap.add_argument("--id", type=int, default=0, help="single: the marker's id")
    ap.add_argument("--ids", type=int, nargs=4, help="board: the 4 ids, TL TR BL BR")
    ap.add_argument("--marker-mm", type=float, help="black edge to black edge")
    ap.add_argument("--pitch-mm", type=float, help="board: centre to centre")
    ap.add_argument("--dict", choices=sorted(DICTS), help="ArUco dictionary (default 4X4_50)")
    ap.add_argument("--pad-drop-mm", type=float, help="target centre down to the pad surface")
    ap.add_argument("--pad-out-mm", type=float, help="target face out to the pad centre")
    ap.add_argument("--tilt-deg", type=float, help="how far it leans back from vertical")
    ap.add_argument("--live", action="store_true", help="identify: one frame from the Tello")
    ap.add_argument("--file", help="board.json to read and write (default: next to the "
                                   f"code, or ${ENV})")
    args = ap.parse_args()
    env = os.environ.get(ENV)
    path = pathlib.Path(args.file or (env if env and env != "default" else PATH))

    if args.kind == "identify":
        if args.live:
            img = _grab_frame()
        elif args.image:
            img = cv2.imread(args.image)
            if img is None:
                raise SystemExit(f"cannot read {args.image}")
        else:
            ap.error("identify needs an image or --live")
        found = identify(img)
        if not found:
            print("no ArUco marker found in any common dictionary - closer, flatter, "
                  "better lit, or it is not an ArUco/AprilTag marker")
            return
        for name, i, side in found:
            print(f"  dictionary {name:14s} id {i:3d}   ({side:.0f} px across)")
        if any(n == "4X4_50" for n, _, _ in found):
            print("-> 4X4_50 is what this project uses by default.")
        else:
            print("-> not 4X4_50: configure it with --dict, e.g. "
                  f"python board_config.py single --id {found[0][1]} --dict {found[0][0]} "
                  "--marker-mm <measured>")
        return

    if args.kind == "reset":
        if path.exists():
            path.unlink()
        print(f"{path} removed - back to the printed 2x2 board")
        print(describe(BoardGeometry()))
        return

    geom = load_board(path) if path.exists() else BoardGeometry()
    if args.kind is not None or any(v is not None for v in (
            args.marker_mm, args.pitch_mm, args.dict, args.pad_drop_mm,
            args.pad_out_mm, args.tilt_deg)):
        kw = {}
        if args.kind == "single":
            kw["ids"] = (args.id,)
            if args.marker_mm is None and not geom.single:
                ap.error("single: give the marker's size, --marker-mm (measure the "
                         "black square, edge to edge)")
        elif args.kind == "board":
            kw["ids"] = tuple(args.ids) if args.ids else (0, 1, 2, 3)
            if args.marker_mm is None and geom.single:
                kw["marker_m"] = BoardGeometry.marker_m     # back to the printed sheet
        if args.marker_mm is not None:
            kw["marker_m"] = args.marker_mm / 1000.0
        if args.pitch_mm is not None:
            kw["pitch_m"] = args.pitch_mm / 1000.0
        if args.dict is not None:
            kw["dict_id"] = DICTS[args.dict]
        if args.pad_drop_mm is not None:
            kw["pad_drop_m"] = args.pad_drop_mm / 1000.0
        if args.pad_out_mm is not None:
            kw["pad_out_m"] = args.pad_out_mm / 1000.0
        if args.tilt_deg is not None:
            kw["tilt_deg"] = args.tilt_deg
        geom = dataclasses.replace(geom, **kw)
        geom.check()
        save_board(geom, path)
        print(f"wrote {path}")
    else:
        where = path if path.exists() else "defaults (no board.json)"
        print(f"target from {where}:")
    print(describe(geom))


if __name__ == "__main__":
    main()
