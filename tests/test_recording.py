"""The GUI's recorder: real-time playback whatever the UI's tick rate, one
CSV row per video frame, fixed frame size, and camera-only recordings
that tello_map.py can replay.

    python tests/test_recording.py
"""

import csv
import os
import pathlib
import sys
import tempfile

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("TELLO_BOARD", "default")   # test the printed 2x2 board, whatever board.json says
sys.path.insert(0, str(ROOT))
from tello_gui import Recorder                      # noqa: E402


def _record(folder, with_panel, ticks, view_size=(960, 720)):
    """Tick i shows a picture of brightness 10*i and logs px = i, so a
    frame and its CSV row can be checked against each other."""
    rec = Recorder(folder, fps=20.0, with_panel=with_panel, stem="t")
    panel = np.full((470, 470, 3), 80, np.uint8)     # a 3D-view-sized panel
    t = 1000.0
    for i, dt in enumerate(ticks):
        view = np.full((view_size[1], view_size[0], 3), (10 * i) % 250, np.uint8)
        row = ["HOLD", "board", f"{i:.1f}", "0.0", "100.0"] + [""] * 19
        rec.write(view, panel, row, now=t)
        t += dt
    last = t - ticks[-1]
    rec.close()
    return rec, last - 1000.0


def test_real_time_with_irregular_ticks():
    # a jittery UI: 30-80 ms ticks and a 400 ms stall
    ticks = [0.03, 0.08, 0.05, 0.4, 0.05, 0.06] * 12
    with tempfile.TemporaryDirectory() as d:
        rec, span = _record(d, True, ticks)
        cap = cv2.VideoCapture(str(rec.video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()
        rows = list(csv.reader(open(rec.csv_path)))
        assert rows[0] == Recorder.COLS
        assert n == len(rows) - 1 == rec.frames, (n, len(rows), rec.frames)
        assert abs(rec.frames / 20.0 - span) <= 0.1, (rec.frames / 20.0, span)
        assert size == (1270, 600), size             # camera 800x600 + panel 470x600


def test_camera_only_is_full_size_and_replayable():
    """Frame k shows what CSV row k logged - the pairing tello_map.py
    relies on - including across a stall, where one picture is repeated."""
    import tello_map
    with tempfile.TemporaryDirectory() as d:
        # recording started before video arrived: placeholder-sized view
        ticks = [0.05] * 10 + [0.3] + [0.05] * 10
        rec, _ = _record(d, False, ticks, view_size=(640, 480))
        cap = cv2.VideoCapture(str(rec.video_path))
        size = (int(cap.get(3)), int(cap.get(4)))
        shown = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            shown.append(int(round(f.mean() / 10.0)))
        cap.release()
        assert size == (960, 720), size
        log = tello_map.read_log(str(rec.csv_path))
        assert len(log) == rec.frames == len(shown), (len(log), rec.frames, len(shown))
        logged = [int(p[0]) for _, p, _, _ in log]
        assert shown == logged, (shown, logged)
        assert logged.count(10) == 6, logged        # the stall: tick 10 held 0.3 s


def main():
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print("\nPASS" if not failed else f"\nFAIL ({failed})")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
