"""
Produce tello_calib.npz (K, dist, and how much to trust them) from
chessboard images.

    python tello_pose.py --live --snap      # press 's' ~25 times
    python calibrate_camera.py              # solve, check, write
    python calibrate_camera.py --check      # re-check an existing tello_calib.npz

Collect frames with the ACTUAL Tello camera - intrinsics from any other
camera are worthless here.

WHAT MAKES A CALIBRATION WORK
    The focal length is only measurable from perspective: a board seen
    square-on at one distance looks exactly like a board of a different size
    at a different distance through a different lens. So the views have to
    DIFFER:
      - tilt the board 20-45 deg, toward and away, left and right
      - vary the distance (near: board fills 2/3 of the frame; far: 1/4)
      - put the board in the corners of the frame, not just the middle
    Move the board, not the drone: hold the drone still on a box at head
    height and walk the board around in front of it.

    The set this repo shipped with had 11 of 12 usable views at the same
    1.65 m and the same ~23 deg (a board propped on the floor, drone on the
    floor, only sliding sideways). The solve returned fx = 1128 +- 54 px -
    about 20% high - and the hop computed from it was long by the same
    fraction. This script now measures that and says so.

WHAT IT DOES
    Fixes k3 (and the aspect ratio: the Tello has square pixels) so a thin
    data set cannot buy a low RMS with a wild lens model, drops frames whose
    error is far above the rest and re-solves, reports each parameter's
    standard deviation and the diversity of the views, and writes all of it
    into the npz - tello_pose.load_calibration() refuses to fly on a file
    that fails the same checks.
"""

import argparse
import glob
import math

import cv2
import numpy as np

INNER = (9, 6)          # inner corners, matches calibration_chessboard_A4.pdf
SQUARE_MM = 25.0


def find_corners(gray, refine=True, fast=False):
    """Chessboard inner corners, or None. fast=True is for live preview -
    FAST_CHECK bails early on frames with no board rather than running the
    full search every video frame. Otherwise the sector-based detector is
    the fallback: slower, but it finds boards the classic one misses (and
    its corners are accurate enough to skip cornerSubPix)."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags += cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, INNER, flags=flags)
    if ok:
        if refine:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        return corners
    if fast:
        return None
    ok, corners = cv2.findChessboardCornersSB(
        gray, INNER, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    return corners.astype(np.float32) if ok else None


def _object_points():
    objp = np.zeros((INNER[0] * INNER[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:INNER[0], 0:INNER[1]].T.reshape(-1, 2)
    return objp * (SQUARE_MM / 1000.0)          # metres


def _view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist):
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        errs.append(float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - img_pts[i].reshape(-1, 2)) ** 2, axis=1)))))
    return errs


def view_diversity(rvecs, tvecs, img_pts, size, grid=6):
    """How different the views are - the thing that decides whether the
    focal length is measured at all."""
    tilts, dirs, ranges = [], [], []
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        n = R[:, 2]                              # board normal in camera frame
        tilts.append(math.degrees(math.acos(min(1.0, abs(float(n[2]))))))
        dirs.append(math.degrees(math.atan2(float(n[1]), float(n[0]))))
        ranges.append(float(np.linalg.norm(tv)))
    W, H = size
    cells = set()
    for c in img_pts:
        for u, v in c.reshape(-1, 2):
            cells.add((min(grid - 1, int(u / W * grid)), min(grid - 1, int(v / H * grid))))
    tilts = np.array(tilts)
    oblique = [d for d, t in zip(dirs, tilts) if t > 15.0]
    # distinct tilt directions among oblique views, in 90-degree sectors
    sectors = {int(((d + 45.0) % 360.0) // 90.0) for d in oblique}
    return dict(tilt_min=float(tilts.min()), tilt_max=float(tilts.max()),
                tilt_median=float(np.median(tilts)), tilt_spread=float(np.ptp(tilts)),
                tilt_directions=len(sectors),
                range_min=min(ranges), range_max=max(ranges),
                range_ratio=max(ranges) / max(min(ranges), 1e-6),
                coverage=len(cells) / float(grid * grid))


def diversity_problems(d):
    out = []
    if d["tilt_spread"] < 20.0:
        out.append(f"tilts only span {d['tilt_spread']:.0f} deg "
                   f"({d['tilt_min']:.0f}-{d['tilt_max']:.0f}): tilt the board "
                   "20-45 deg in different directions")
    elif d["tilt_directions"] < 2:
        out.append("every oblique view leans the same way - tilt it the "
                   "other way too (left/right AND up/down)")
    if d["range_ratio"] < 1.5:
        out.append(f"all views at {d['range_min']:.2f}-{d['range_max']:.2f} m: "
                   "vary the distance (board filling 2/3 of the frame, then 1/4)")
    if d["coverage"] < 0.6:
        out.append(f"corners cover only {d['coverage']*100:.0f}% of the image: "
                   "put the board in the corners and edges of the frame")
    return out


def calibrate(files, on_frame=None, fix_k3=True, fix_aspect=True, reject=True):
    """Full solve: returns a dict with rms, K, dist, size, errs, used,
    std (fx, fy, cx, cy), diversity, problems."""
    objp = _object_points()
    obj_pts, img_pts, size, used = [], [], None, []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_corners(gray)
        if on_frame:
            on_frame(f, corners)
        if corners is None:
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(f)
    if len(used) < 10:
        raise ValueError(f"only {len(used)} usable frames - "
                         "need at least 10, ideally 20-30")

    flags = 0
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    K0 = None
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
        W, H = size
        K0 = np.array([[900.0, 0, W / 2.0], [0, 900.0, H / 2.0], [0, 0, 1.0]])

    def solve(O, I):
        K_init = None if K0 is None else K0.copy()
        rms, K, dist, rv, tv, sdi, _, _ = cv2.calibrateCameraExtended(
            O, I, size, K_init, None, flags=flags)
        sd = sdi.ravel().copy()
        if fix_aspect:
            sd[0] = sd[1]          # fx is tied to fy, and inherits its doubt
        return rms, K, dist, rv, tv, sd

    rms, K, dist, rvecs, tvecs, std = solve(obj_pts, img_pts)
    errs = _view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)
    dropped = []
    if reject and len(errs) > 12:
        # One bad frame (motion blur, a bent sheet, a misdetected corner)
        # drags the whole fit; it shows up here rather than in the RMS.
        lim = max(3.0 * float(np.median(errs)), 1.0)
        keep = [i for i, e in enumerate(errs) if e <= lim]
        if len(keep) < len(errs) and len(keep) >= 10:
            dropped = [used[i] for i in range(len(used)) if i not in keep]
            obj_pts = [obj_pts[i] for i in keep]
            img_pts = [img_pts[i] for i in keep]
            used = [used[i] for i in keep]
            rms, K, dist, rvecs, tvecs, std = solve(obj_pts, img_pts)
            errs = _view_errors(obj_pts, img_pts, rvecs, tvecs, K, dist)

    from tello_pose import calibration_problems
    div = view_diversity(rvecs, tvecs, img_pts, size)
    problems = calibration_problems(K, dist, size, rms, std[:4])
    problems += diversity_problems(div)
    return dict(rms=float(rms), K=K, dist=dist, size=size, errs=errs, used=used,
                dropped=dropped, std=std[:4], diversity=div, problems=problems)


def solve_calibration(files, on_frame=None):
    """(rms, K, dist, size, errs, used) - the older tuple form, kept for
    callers that only want the numbers."""
    r = calibrate(files, on_frame)
    return r["rms"], r["K"], r["dist"], r["size"], r["errs"], r["used"]


def save_calibration(path, r):
    np.savez(path, K=r["K"], dist=r["dist"], rms=r["rms"],
             image_size=np.array(r["size"]), std_intrinsics=np.asarray(r["std"]),
             n_frames=len(r["used"]), problems=np.array(r["problems"], dtype=str))


def fov_deg(K, size):
    return (2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0]))),
            2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1]))))


def report(r):
    K, std, d = r["K"], r["std"], r["diversity"]
    fov_x, fov_y = fov_deg(K, r["size"])
    print(f"\nusable frames: {len(r['used'])}"
          + (f"  (dropped {len(r['dropped'])} outlier(s): "
             f"{', '.join(r['dropped'])})" if r["dropped"] else ""))
    print(f"image size: {r['size']}")
    print(f"overall RMS: {r['rms']:.4f} px   (want < 0.5; > 1.0 means recollect)")
    print(f"fx {K[0,0]:7.1f} +- {std[0]:5.1f}   fy {K[1,1]:7.1f} +- {std[1]:5.1f}")
    print(f"cx {K[0,2]:7.1f} +- {std[2]:5.1f}   cy {K[1,2]:7.1f} +- {std[3]:5.1f}   px")
    print(f"dist {np.round(np.asarray(r['dist']).ravel(), 4)}")
    print(f"FOV: {fov_x:.1f} x {fov_y:.1f} deg   (a Tello stream is ~55 x 43)")
    print(f"views: tilt {d['tilt_min']:.0f}-{d['tilt_max']:.0f} deg "
          f"({d['tilt_directions']} direction(s)), range {d['range_min']:.2f}-"
          f"{d['range_max']:.2f} m, image coverage {d['coverage']*100:.0f}%")
    order = np.argsort(r["errs"])[::-1]
    print("worst frames:")
    for i in order[:5]:
        print(f"  {r['errs'][i]:6.3f} px  {r['used'][i]}")
    if r["problems"]:
        print("\n!! NOT GOOD ENOUGH TO FLY ON:")
        for p in r["problems"]:
            print(f"   - {p}")
    else:
        print("\nlooks good - validate with a tape measure before flying "
              "(README step 4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default="calib/*.png",
                    help="glob, default calib/*.png")
    ap.add_argument("--out", default="tello_calib.npz")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--free-k3", action="store_true",
                    help="let k3 float (only with lots of wide-angle coverage)")
    ap.add_argument("--free-aspect", action="store_true",
                    help="solve fx and fy independently")
    ap.add_argument("--check", action="store_true",
                    help="only check an existing --out file")
    args = ap.parse_args()

    if args.check:
        from tello_pose import load_calibration
        K, dist, ok = load_calibration(args.out)
        print(f"fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f} "
              f"dist {np.round(dist.ravel(), 4)}")
        print("OK to fly on" if ok else "NOT OK to fly on")
        raise SystemExit(0 if ok else 1)

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files match {args.pattern}")

    def on_frame(f, corners):
        if corners is None:
            print(f"  skip {f} (no board found)")
        elif args.show:
            img = cv2.imread(f)
            cv2.drawChessboardCorners(img, INNER, corners, True)
            cv2.imshow("corners", img)
            cv2.waitKey(120)

    try:
        r = calibrate(files, on_frame, fix_k3=not args.free_k3,
                      fix_aspect=not args.free_aspect)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.show:
        cv2.destroyAllWindows()
    report(r)
    save_calibration(args.out, r)
    print(f"\nwrote {args.out}")
    raise SystemExit(1 if r["problems"] else 0)


if __name__ == "__main__":
    main()
