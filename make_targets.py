"""Generate 1:1 print-scale PDFs: the landing target, the calibration
chessboard, and standalone room markers for room_map.py / room_pose.py.

    python make_targets.py                                  # -> print/
    python make_targets.py --room-markers 8 --room-marker-mm 150

The landing target is whatever board_config.py has configured (board.json;
default: the 2x2 board) - the same geometry the estimator, the controller
and the simulator use - so the printed sheet and the code cannot drift
apart. For a single big marker instead of the board:

    python board_config.py single --id 0 --marker-mm 160
    python make_targets.py          # -> print/landing_marker_id00_160mm_A4.pdf
"""

import argparse
import pathlib

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import cv2
from board_config import load_board
from room_map import ROOM_ID_MIN

# ---- target geometry: derived from board.json / BoardGeometry, never restated ----
GEOM = load_board()
DICT_ID = GEOM.dict_id
MARKER_MM = GEOM.marker_m * 1000.0
PITCH_MM = GEOM.pitch_m * 1000.0     # centre-to-centre (2x2 board only)
IDS = {mid: (int(np.sign(cx)), int(np.sign(cy)))      # (x, y) signs; (0, 0) for one marker
       for mid, (cx, cy) in GEOM.layout.items()}

# ---- room markers (must match --marker-size in map_room.py) ----
ROOM_MARKER_MM = 150.0               # bigger than the landing board's 70mm:
                                     # these need to read at room, not standoff, range

CELL_MM = MARKER_MM / 6.0            # 4x4 dict => 6x6 cells incl. black border
GAP_MM = PITCH_MM - MARKER_MM
SPAN_MM = GEOM.span_m * 1000.0

if not GEOM.single:
    assert GAP_MM / 2 >= CELL_MM, "gap too small for a one-cell quiet zone"
# the white margin around the outside must be a cell wide too, on A4 portrait
assert SPAN_MM + 2 * CELL_MM <= 210.0, (
    f"a {SPAN_MM:.0f} mm target does not fit A4 with its quiet zone - at most "
    f"{210.0 - 2 * CELL_MM:.0f} mm")

PX_PER_MARKER = 1200                 # ~435 dpi at 70 mm

ORIGIN_MM = (105.0, 148.5)           # board origin, centred on A4 portrait


def marker_image(mid):
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    img = cv2.aruco.generateImageMarker(d, mid, PX_PER_MARKER)
    return ImageReader(Image.fromarray(img).convert("L"))


def _top_mark(c, text):
    """'this way up', in grey, in the top margin - 5 cm clear of any
    marker, far outside every quiet zone. The sheets used to carry no marks
    at all, and the landing board was mounted upside down in every recorded
    flight: that negates x and y, and the drone flies away from the pad."""
    x = ORIGIN_MM[0] * mm
    y = 282.0 * mm
    c.setFillGray(0.45)
    p = c.beginPath()
    p.moveTo(x, y + 6 * mm)
    p.lineTo(x - 3.5 * mm, y + 1 * mm)
    p.lineTo(x + 3.5 * mm, y + 1 * mm)
    p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x, y - 3.5 * mm, "TOP")
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(x, y - 7.5 * mm, text)
    c.setFillGray(0.0)


def _scale_bar(c, length_mm=100.0):
    """A printed-scale check that needs only a ruler: the bar must measure
    exactly length_mm. If it does not, the printer scaled the page."""
    x0 = (ORIGIN_MM[0] - length_mm / 2) * mm
    y = 16.0 * mm
    c.setStrokeGray(0.45)
    c.setLineWidth(0.6)
    c.line(x0, y, x0 + length_mm * mm, y)
    for k in range(int(length_mm / 10) + 1):
        h = 2.5 if k % 5 else 4.0
        c.line(x0 + k * 10 * mm, y, x0 + k * 10 * mm, y + h * mm)
    c.setFillGray(0.45)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(ORIGIN_MM[0] * mm, y - 4.5 * mm,
                        f"this bar must measure {length_mm:.0f} mm - print at 100%, "
                        "never 'fit to page'")
    c.setFillGray(0.0)


def build_board(path, marks=True):
    c = canvas.Canvas(path, pagesize=A4)
    cx_mm, cy_mm = ORIGIN_MM
    for mid, (sx, sy) in IDS.items():
        mx = cx_mm + sx * PITCH_MM / 2.0
        my = cy_mm + sy * PITCH_MM / 2.0
        c.drawImage(marker_image(mid),
                    (mx - MARKER_MM / 2) * mm, (my - MARKER_MM / 2) * mm,
                    width=MARKER_MM * mm, height=MARKER_MM * mm)
    if marks:
        if GEOM.single:
            _top_mark(c, f"landing marker id {GEOM.ids[0]} - {MARKER_MM:.0f} mm - "
                         "mount vertical, flat, this edge up")
        else:
            _top_mark(c, f"landing board - marker id {GEOM.ids[0]} top-left - "
                         f"{MARKER_MM:.0f} mm markers - mount vertical, flat")
        _scale_bar(c)
    c.showPage()
    c.save()


def build_chessboard(path, cols=10, rows=7, sq_mm=25.0):
    """cols x rows squares -> (cols-1) x (rows-1) inner corners."""
    pw, ph = landscape(A4)
    c = canvas.Canvas(path, pagesize=landscape(A4))
    w_mm, h_mm = cols * sq_mm, rows * sq_mm
    x0 = (pw / mm - w_mm) / 2.0
    y0 = (ph / mm - h_mm) / 2.0
    c.setFillGray(0.0)
    for r in range(rows):
        for col in range(cols):
            if (r + col) % 2 == 0:
                c.rect((x0 + col * sq_mm) * mm, (y0 + r * sq_mm) * mm,
                       sq_mm * mm, sq_mm * mm, stroke=0, fill=1)
    c.showPage()
    c.save()


def build_room_marker(path, mid, size_mm=ROOM_MARKER_MM, marks=True):
    """One marker, centred. With marks: its id, size and which way is up
    in the top margin - room markers must be mounted upright too (room +Y
    is taken as vertical by everything that flies on the map)."""
    c = canvas.Canvas(path, pagesize=A4)
    cx_mm, cy_mm = ORIGIN_MM
    c.drawImage(marker_image(mid),
                (cx_mm - size_mm / 2) * mm, (cy_mm - size_mm / 2) * mm,
                width=size_mm * mm, height=size_mm * mm)
    if marks:
        _top_mark(c, f"room marker id {mid} - {size_mm:.0f} mm "
                     f"(map_room.py --marker-size {size_mm / 1000:.3f})")
    c.showPage()
    c.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="regenerate the printable targets")
    ap.add_argument("--out-dir",
                    default=str(pathlib.Path(__file__).parent / "print"))
    ap.add_argument("--room-markers", type=int, default=8,
                    help="how many room markers to generate, ids "
                         f"{ROOM_ID_MIN}.. (default 8)")
    ap.add_argument("--room-marker-mm", type=float, default=ROOM_MARKER_MM,
                    help="room marker edge length in mm; pass the same size "
                         "to map_room.py's --marker-size in METRES "
                         f"(default {ROOM_MARKER_MM:.0f} mm = "
                         f"{ROOM_MARKER_MM / 1000:.2f})")
    ap.add_argument("--no-marks", action="store_true",
                    help="no TOP mark / scale bar / labels in the margins")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marks = not args.no_marks
    if GEOM.single:
        target = out / f"landing_marker_id{GEOM.ids[0]:02d}_{MARKER_MM:.0f}mm_A4.pdf"
    else:
        target = out / "aruco_landing_board_A4.pdf"
    build_board(str(target), marks)
    print(f"landing target: {target.name}  ({GEOM.check()})")
    build_chessboard(str(out / "calibration_chessboard_A4.pdf"))
    room_ids = range(ROOM_ID_MIN, ROOM_ID_MIN + args.room_markers)
    for mid in room_ids:
        build_room_marker(str(out / f"room_marker_{mid:02d}_A4.pdf"),
                          mid, args.room_marker_mm, marks)
    print(f"wrote {2 + args.room_markers} PDFs to {out}/")
    print(f"marker {MARKER_MM} mm | cell {CELL_MM:.2f} mm | gap {GAP_MM} mm | "
          f"span {SPAN_MM} mm | origin {ORIGIN_MM} mm")
    print(f"room markers: ids {list(room_ids)} at {args.room_marker_mm} mm "
          "each - pass the same size to map_room.py's --marker-size")
