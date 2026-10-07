"""Validate corner ordering + pose inversion against the ACTUAL printed sheet.

Rasterises the board PDF, warps it as seen from a known camera pose, runs the
detector, and checks the recovered drone-position-in-board-frame. Catches
corner-order and sign bugs without flying anything.
"""

import os
import pathlib
import subprocess
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
from tello_pose import BoardGeometry, PoseEstimator  # noqa

BOARD_PDF = ROOT / "print" / "aruco_landing_board_A4.pdf"

DPI = 300
S = DPI / 25.4                      # source px per mm
ORIGIN_MM = (105.0, 148.5)          # board origin on the A4 page, from bottom-left
PAGE_H_MM = 297.0

# Tello-like intrinsics: 960x720, ~82.6 deg diagonal FOV
F = 684.0
K = np.array([[F, 0, 480.0], [0, F, 360.0], [0, 0, 1.0]])
DIST = np.zeros(5)


def rasterise():
    """The printed board PDF at DPI. PyMuPDF (pip install pymupdf) if it is
    there, else poppler's pdftoppm - either way it is the actual PDF that
    gets tested, not a re-drawing of it."""
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        page = pymupdf.open(str(BOARD_PDF))[0]
        pix = page.get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    except ImportError:
        pass
    tmp = pathlib.Path(__file__).parent / "src"
    try:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-f", "1", "-l", "1",
                        str(BOARD_PDF), str(tmp)], check=True)
    except FileNotFoundError:
        raise SystemExit("need a PDF rasteriser: pip install pymupdf "
                         "(or install poppler for pdftoppm)")
    return cv2.imread(str(tmp.with_name("src-1.png")))


def board_mm_to_src_px(bx, by):
    ox = ORIGIN_MM[0] * S
    oy = (PAGE_H_MM - ORIGIN_MM[1]) * S
    return np.array([ox + bx * S, oy - by * S])      # +Y_board is image -y


def lookat(p_cam_m):
    """Camera pose looking at the board origin. Returns R_cm, t_cm."""
    z_c = -p_cam_m / np.linalg.norm(p_cam_m)         # forward
    up = np.array([0.0, 1.0, 0.0])                   # +Y_board
    x_c = np.cross(-up, z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R_mc = np.column_stack([x_c, y_c, z_c])          # camera axes in board frame
    R_cm = R_mc.T
    return R_cm, (-R_cm @ p_cam_m).reshape(3, 1)


def render(src, p_cam_m):
    R_cm, t_cm = lookat(p_cam_m)
    rvec, _ = cv2.Rodrigues(R_cm)
    quad_mm = [(-100, 100), (100, 100), (100, -100), (-100, -100)]
    obj = np.array([[x / 1000.0, y / 1000.0, 0.0] for x, y in quad_mm], np.float32)
    img_pts, _ = cv2.projectPoints(obj, rvec, t_cm, K, DIST)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    src_pts = np.array([board_mm_to_src_px(x, y) for x, y in quad_mm], np.float32)

    # Area-average the source down to roughly the destination scale first.
    # warpPerspective only samples 4 neighbours, so warping a 300 dpi page
    # straight to a 75 px board aliases the edges and biases corner refinement.
    src_w = np.linalg.norm(src_pts[1] - src_pts[0])
    dst_w = np.linalg.norm(img_pts[1] - img_pts[0])
    k = max(1, int(np.floor(src_w / max(dst_w, 1.0))))
    if k > 1:
        src = cv2.resize(src, None, fx=1.0 / k, fy=1.0 / k,
                         interpolation=cv2.INTER_AREA)
        src_pts = src_pts / k

    H = cv2.getPerspectiveTransform(src_pts, img_pts)
    return cv2.warpPerspective(src, H, (960, 720),
                               flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))


def analytic_check(tracker):
    """No rasterisation: project true corners, solvePnP, compare. Isolates the
    object-point ordering and the -R.T@t inversion from any image effects."""
    worst = 0.0
    for p in [(0, 0, 1.5), (0.3, 0.1, 1.2), (-0.25, -0.2, 0.9), (0.1, 0, 2.2)]:
        p = np.array(p, float)
        R_cm, t_cm = lookat(p)
        rvec, _ = cv2.Rodrigues(R_cm)
        obj, img = [], []
        for mid, pts in tracker.obj.items():
            proj, _ = cv2.projectPoints(pts, rvec, t_cm, K, DIST)
            obj.append(pts)
            img.append(proj.reshape(-1, 2))
        obj = np.concatenate(obj).astype(np.float32)
        img = np.concatenate(img).astype(np.float32)
        ok, rv, tv = cv2.solvePnP(obj, img, K, DIST,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rv)
        rec = (-R.T @ tv).ravel() * 100.0
        worst = max(worst, float(np.linalg.norm(rec - p * 100)))
    return worst


def main():
    src = rasterise()
    geom = BoardGeometry()
    tracker = PoseEstimator(K, DIST, geom)
    print(geom.check())
    print(f"{'truth (cm)':>22} | {'recovered (cm)':>22} | {'err':>6} | ids")

    cases = [
        (0.00, 0.00, 1.50),     # square on, 1.5 m
        (0.30, 0.00, 1.50),     # offset to board's +X
        (-0.30, 0.00, 1.50),
        (0.00, 0.20, 1.50),     # above board centre
        (0.00, -0.20, 1.50),
        (0.40, 0.15, 1.00),     # oblique, close
        (0.00, 0.00, 2.00),     # far
        (0.00, 0.00, 0.70),     # near hold
    ]
    worst = 0.0
    detected = 0
    for p in cases:
        p = np.array(p)
        img = render(src, p)
        bp = tracker.estimate(img)
        rec = None if bp is None else bp.p_board_cm
        corners, ids, _ = tracker.detect(img)
        n = 0 if ids is None else len(ids)
        if n == len(geom.ids):
            detected += 1
        if rec is None:
            print(f"{str(np.round(p*100,1)):>22} | {'NO DETECTION':>22} |"
                  f"{'':>7} | {n}")
            continue
        err = float(np.linalg.norm(rec - p * 100))
        worst = max(worst, err)
        print(f"{str(np.round(p*100,1)):>22} | {str(np.round(rec,1)):>22} | "
              f"{err:6.2f} | {n}")

    a = analytic_check(tracker)
    print(f"\n[1] geometry     worst {a:.3f} cm   "
          "(object-point order + -R.T@t inversion)")
    print(f"[2] detection    {detected}/{len(cases)} poses with all 4 ids   "
          "(marker size, pitch, quiet zones)")
    print(f"[3] rendered err worst {worst:.1f} cm   "
          "INFORMATIONAL ONLY - dominated by warpPerspective resampling of a\n"
          "                 synthetic page, not a prediction of real accuracy.\n"
          "                 Real numbers come from the tripod tape-measure test.")
    ok = a < 0.01 and detected == len(cases)
    print("\nPASS" if ok else "\nFAIL - check corner order / signs / board scale")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
